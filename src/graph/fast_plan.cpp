#include "fast_plan.hpp"

#include "marmot/error.h"
#include "marmot/op_metadata.gen.h"

#include <algorithm>
#include <string_view>
#include <utility>

#include "graph_handle.hpp"
#include "graph_impl.hpp"
#include "graph_node.hpp"

namespace marmot::graph {

class FastPlanCompiler {
  public:
    [[nodiscard]] static std::expected<FastPlan, marmot_error_t>
    compile(const marmot_graph_t *graph, const FastPlanBucket &bucket);

  private:
    static void push_stage(
        FastPlan &plan, FastStageKind kind, FastStageLowering lowering, std::span<const uint8_t> lowered_node_offsets,
        uint32_t first_node, uint32_t node_count, bool supports_fast_exec, bool is_boundary, FastStagePayload payload
    );
};

namespace {

[[nodiscard]] FastPlanPhase infer_phase(const FastPlanBucket &bucket) noexcept {
    if (bucket.token_count == 0) {
        return FastPlanPhase::Unknown;
    }
    if (!bucket.emit_logits || bucket.sample_count == 0) {
        return FastPlanPhase::Prefill;
    }
    if (bucket.sample_count == bucket.token_count && bucket.token_count <= 8) {
        return FastPlanPhase::Decode;
    }
    if (bucket.sample_count < bucket.token_count) {
        return FastPlanPhase::Hybrid;
    }
    return FastPlanPhase::Decode;
}

[[nodiscard]] bool is_attention_op(marmot_op_id_t op_id) noexcept {
    return op_id == MARMOT_OP_PAGED_ATTENTION;
}

[[nodiscard]] bool is_reshape_op(marmot_op_id_t op_id) noexcept {
    return op_id == MARMOT_OP_RESHAPE;
}

[[nodiscard]] bool is_moe_op(marmot_op_id_t op_id) noexcept {
    return op_id == MARMOT_OP_MOE_EXPERTS;
}

[[nodiscard]] bool is_rms_norm_op(marmot_op_id_t op_id) noexcept {
    return op_id == MARMOT_OP_RMS_NORM || op_id == MARMOT_OP_RMS_NORM_GEMMA;
}

[[nodiscard]] bool is_residual_add_op(marmot_op_id_t op_id) noexcept {
    return op_id == MARMOT_OP_ADD;
}

[[nodiscard]] bool is_ffn_compute_op(marmot_op_id_t op_id) noexcept {
    return op_id == MARMOT_OP_MATMUL || op_id == MARMOT_OP_VIEW || op_id == MARMOT_OP_SLICE ||
        op_id == MARMOT_OP_SWIGLU || op_id == MARMOT_OP_GEGLU || op_id == MARMOT_OP_GELU || op_id == MARMOT_OP_TOPK ||
        op_id == MARMOT_OP_SOFTMAX || op_id == MARMOT_OP_CONVERT || op_id == MARMOT_OP_MOE_EXPERTS;
}

[[nodiscard]] bool is_logits_tail_op(marmot_op_id_t op_id) noexcept {
    return op_id == MARMOT_OP_GATHER_ROWS || op_id == MARMOT_OP_VEC_DOT || op_id == MARMOT_OP_MATMUL ||
        op_id == MARMOT_OP_SOFTMAX;
}

[[nodiscard]] bool has_fast_stage_hint(const GraphNode &node) noexcept {
    return node.fast_stage_hint != MARMOT_FAST_STAGE_HINT_NONE;
}

[[nodiscard]] FastStageKind fast_stage_kind_from_hint(marmot_fast_stage_hint_t stage_hint) noexcept {
    switch (stage_hint) {
    case MARMOT_FAST_STAGE_HINT_ATTENTION:
        return FastStageKind::Attention;
    case MARMOT_FAST_STAGE_HINT_DENSE_FFN:
        return FastStageKind::DenseFfn;
    case MARMOT_FAST_STAGE_HINT_MOE_FFN:
        return FastStageKind::MoeFfn;
    case MARMOT_FAST_STAGE_HINT_LOGITS_TAIL:
        return FastStageKind::LogitsTail;
    case MARMOT_FAST_STAGE_HINT_NONE:
    case MARMOT_FAST_STAGE_HINT_COUNT:
        break;
    }
    return FastStageKind::GenericFallback;
}

[[nodiscard]] bool uses_annotated_roles(std::span<const GraphNode> nodes) noexcept {
    return std::any_of(nodes.begin(), nodes.end(), [](const GraphNode &node) {
        return node.fast_node_role != MARMOT_FAST_NODE_ROLE_NONE;
    });
}

[[nodiscard]] const GraphNode *find_role_node(std::span<const GraphNode> nodes, marmot_fast_node_role_t role) noexcept {
    const GraphNode *result = nullptr;
    for (const GraphNode &node : nodes) {
        if (node.fast_node_role != role) {
            continue;
        }
        if (result != nullptr) {
            return nullptr;
        }
        result = &node;
    }
    return result;
}

struct StageMatch {
    FastStageKind kind{FastStageKind::GenericFallback};
    FastStageLowering lowering{FastStageLowering::None};
    uint32_t node_count{0};
    std::array<uint8_t, 8> lowered_node_offsets{};
    uint8_t lowered_node_count{0};
};

[[nodiscard]] StageMatch make_stage_match(
    FastStageKind kind, FastStageLowering lowering, uint32_t node_count, std::span<const uint8_t> lowered_node_offsets
) noexcept {
    StageMatch match{
        .kind = kind,
        .lowering = lowering,
        .node_count = node_count,
    };
    const size_t count = std::min(match.lowered_node_offsets.size(), lowered_node_offsets.size());
    for (size_t i = 0; i < count; ++i) {
        match.lowered_node_offsets[i] = lowered_node_offsets[i];
    }
    match.lowered_node_count = static_cast<uint8_t>(count);
    return match;
}

[[nodiscard]] FastBytecodeOpRef make_bc_op_ref(const GraphNode &node) noexcept {
    return FastBytecodeOpRef{
        .signature = node.signature,
        .bc_op_index = node.bc_op_index,
    };
}

[[nodiscard]] std::optional<FastRmsNormOp> build_rms_norm_op(const GraphNode &node) noexcept {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return std::nullopt;
    }
    return FastRmsNormOp{
        .op = make_bc_op_ref(node),
        .input = node.inputs[0],
        .weight = node.inputs[1],
        .output = node.outputs[0],
    };
}

[[nodiscard]] std::optional<FastMatmulOp> build_matmul_op(const GraphNode &node) noexcept {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return std::nullopt;
    }
    FastMatmulOp op{
        .op = make_bc_op_ref(node),
        .input = node.inputs[0],
        .weight = node.inputs[1],
        .output = node.outputs[0],
    };
    if ((node.signature.epilogue_flags & MARMOT_EPILOGUE_BIAS) != 0) {
        if (node.inputs.size() < 3) {
            return std::nullopt;
        }
        op.bias = node.inputs[2];
    }
    return op;
}

[[nodiscard]] std::optional<FastUnaryOp> build_unary_op(const GraphNode &node) noexcept {
    if (node.inputs.empty() || node.outputs.empty()) {
        return std::nullopt;
    }
    return FastUnaryOp{
        .op = make_bc_op_ref(node),
        .input = node.inputs[0],
        .output = node.outputs[0],
    };
}

[[nodiscard]] std::optional<FastBinaryOp> build_binary_op(const GraphNode &node) noexcept {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return std::nullopt;
    }
    return FastBinaryOp{
        .op = make_bc_op_ref(node),
        .input_a = node.inputs[0],
        .input_b = node.inputs[1],
        .output = node.outputs[0],
    };
}

[[nodiscard]] std::optional<FastSoftmaxOp> build_softmax_op(const GraphNode &node) noexcept {
    if (node.inputs.empty() || node.outputs.empty()) {
        return std::nullopt;
    }
    return FastSoftmaxOp{
        .op = make_bc_op_ref(node),
        .input = node.inputs[0],
        .output = node.outputs[0],
        .axis = -1,
    };
}

[[nodiscard]] std::optional<FastTopkOp>
build_topk_op(const GraphNode &node, std::span<const GraphValue> values) noexcept {
    if (node.inputs.size() != 1 || node.outputs.size() != 2) {
        return std::nullopt;
    }
    uint32_t k = 0;
    if (node.outputs[0] < values.size()) {
        const auto &desc = values[node.outputs[0]].desc;
        if (desc.ndim != 0) {
            const size_t width = desc.shape[desc.ndim - 1];
            if (width <= UINT32_MAX) {
                k = static_cast<uint32_t>(width);
            }
        }
    }
    return FastTopkOp{
        .op = make_bc_op_ref(node),
        .input = node.inputs[0],
        .values_out = node.outputs[0],
        .indices_out = node.outputs[1],
        .axis = -1,
        .k = k,
    };
}

[[nodiscard]] std::optional<FastMoeExpertsOp> build_moe_op(const GraphNode &node) noexcept {
    if (node.inputs.size() != 6 || node.outputs.size() != 1) {
        return std::nullopt;
    }
    return FastMoeExpertsOp{
        .op = make_bc_op_ref(node),
        .hidden_states = node.inputs[0],
        .gate_exps = node.inputs[1],
        .up_exps = node.inputs[2],
        .down_exps = node.inputs[3],
        .topk_ids = node.inputs[4],
        .topk_weights = node.inputs[5],
        .output = node.outputs[0],
        .ffn_type = node.moe_ffn_type,
        .weights_scale = node.moe_weights_scale,
        .router_weight_policy = node.moe_router_weight_policy,
    };
}

[[nodiscard]] std::optional<FastGatherRowsOp> build_gather_rows_op(const GraphNode &node) noexcept {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return std::nullopt;
    }
    return FastGatherRowsOp{
        .op = make_bc_op_ref(node),
        .input = node.inputs[0],
        .indices = node.inputs[1],
        .output = node.outputs[0],
    };
}

[[nodiscard]] std::optional<FastVecDotOp> build_vec_dot_op(const GraphNode &node) noexcept {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return std::nullopt;
    }
    return FastVecDotOp{
        .op = make_bc_op_ref(node),
        .input = node.inputs[0],
        .weight = node.inputs[1],
        .output = node.outputs[0],
    };
}

[[nodiscard]] std::optional<FastReshapeOp>
build_reshape_op(const GraphNode &node, std::span<const GraphValue> values) noexcept {
    if (node.inputs.empty() || node.outputs.empty() || node.outputs[0] >= values.size()) {
        return std::nullopt;
    }
    return FastReshapeOp{
        .input = node.inputs[0],
        .output = node.outputs[0],
        .output_desc = values[node.outputs[0]].desc,
    };
}

[[nodiscard]] std::optional<FastRopeOp> build_rope_op(const GraphNode &node) noexcept {
    if (node.inputs.size() < 2 || node.outputs.empty()) {
        return std::nullopt;
    }
    return FastRopeOp{
        .op = make_bc_op_ref(node),
        .input = node.inputs[0],
        .positions = node.inputs[1],
        .output = node.outputs[0],
    };
}

[[nodiscard]] std::optional<FastPagedAttentionOp> build_paged_attention_op(const GraphNode &node) noexcept {
    if (node.inputs.size() < 7 || node.outputs.empty()) {
        return std::nullopt;
    }
    FastPagedAttentionOp op{
        .op = make_bc_op_ref(node),
        .token_meta = node.inputs[0],
        .q = node.inputs[1],
        .k = node.inputs[2],
        .v = node.inputs[3],
        .kv_k = node.inputs[4],
        .kv_v = node.inputs[5],
        .block_table = node.inputs[6],
        .output = node.outputs[0],
        .layer_idx = node.paged_attention_layer_idx,
    };
    if (node.inputs.size() >= 9) {
        op.kv_k_scale = node.inputs[7];
        op.kv_v_scale = node.inputs[8];
    }
    return op;
}

[[nodiscard]] std::optional<FastAttentionDecodeSupernode> build_attention_decode_supernode_annotated(
    std::span<const GraphNode> nodes, std::span<const GraphValue> values
) noexcept {
    const GraphNode *attn_norm_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_NORM);
    const GraphNode *q_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_PROJ);
    const GraphNode *k_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_PROJ);
    const GraphNode *v_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_V_PROJ);
    const GraphNode *q_heads_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_HEADS);
    const GraphNode *k_heads_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_HEADS);
    const GraphNode *v_heads_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_V_HEADS);
    const GraphNode *q_rope_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_ROPE);
    const GraphNode *k_rope_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_ROPE);
    const GraphNode *q_tokens_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_TOKENS);
    const GraphNode *k_tokens_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_TOKENS);
    const GraphNode *v_tokens_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_V_TOKENS);
    const GraphNode *paged_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_PAGED);
    const GraphNode *attn_flat_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_OUT_RESHAPE);
    const GraphNode *out_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_OUT_PROJ);
    const GraphNode *residual_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_RESIDUAL);
    if (attn_norm_node == nullptr || q_proj_node == nullptr || k_proj_node == nullptr || v_proj_node == nullptr ||
        q_heads_node == nullptr || k_heads_node == nullptr || v_heads_node == nullptr || q_rope_node == nullptr ||
        k_rope_node == nullptr || q_tokens_node == nullptr || k_tokens_node == nullptr || v_tokens_node == nullptr ||
        paged_node == nullptr || attn_flat_node == nullptr || out_proj_node == nullptr || residual_node == nullptr) {
        return std::nullopt;
    }

    const auto attn_norm = build_rms_norm_op(*attn_norm_node);
    const auto q_proj = build_matmul_op(*q_proj_node);
    const auto k_proj = build_matmul_op(*k_proj_node);
    const auto v_proj = build_matmul_op(*v_proj_node);
    const auto q_heads = build_reshape_op(*q_heads_node, values);
    const auto k_heads = build_reshape_op(*k_heads_node, values);
    const auto v_heads = build_reshape_op(*v_heads_node, values);
    const auto q_rope = build_rope_op(*q_rope_node);
    const auto k_rope = build_rope_op(*k_rope_node);
    const auto q_tokens = build_reshape_op(*q_tokens_node, values);
    const auto k_tokens = build_reshape_op(*k_tokens_node, values);
    const auto v_tokens = build_reshape_op(*v_tokens_node, values);
    const auto paged_attention = build_paged_attention_op(*paged_node);
    const auto attn_flat = build_reshape_op(*attn_flat_node, values);
    const auto out_proj = build_matmul_op(*out_proj_node);
    const auto residual_add = build_binary_op(*residual_node);
    if (!attn_norm || !q_proj || !k_proj || !v_proj || !q_heads || !k_heads || !v_heads || !q_rope || !k_rope ||
        !q_tokens || !k_tokens || !v_tokens || !paged_attention || !attn_flat || !out_proj || !residual_add) {
        return std::nullopt;
    }

    FastAttentionDecodeSupernode supernode{
        .attn_norm = *attn_norm,
        .q_proj = *q_proj,
        .k_proj = *k_proj,
        .v_proj = *v_proj,
        .q_heads = *q_heads,
        .k_heads = *k_heads,
        .v_heads = *v_heads,
        .q_rope = *q_rope,
        .k_rope = *k_rope,
        .q_tokens = *q_tokens,
        .k_tokens = *k_tokens,
        .v_tokens = *v_tokens,
        .paged_attention = *paged_attention,
        .attn_flat = *attn_flat,
        .out_proj = *out_proj,
        .residual_add = *residual_add,
    };

    if (const GraphNode *q_norm_reshape_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_NORM_RESHAPE)) {
        const GraphNode *q_norm_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_NORM);
        if (q_norm_node == nullptr) {
            return std::nullopt;
        }
        const auto q_norm_reshape = build_reshape_op(*q_norm_reshape_node, values);
        const auto q_norm = build_rms_norm_op(*q_norm_node);
        if (!q_norm_reshape || !q_norm) {
            return std::nullopt;
        }
        supernode.q_norm_reshape = *q_norm_reshape;
        supernode.q_norm = *q_norm;
    } else if (find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_NORM) != nullptr) {
        return std::nullopt;
    }

    if (const GraphNode *k_norm_reshape_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_NORM_RESHAPE)) {
        const GraphNode *k_norm_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_NORM);
        if (k_norm_node == nullptr) {
            return std::nullopt;
        }
        const auto k_norm_reshape = build_reshape_op(*k_norm_reshape_node, values);
        const auto k_norm = build_rms_norm_op(*k_norm_node);
        if (!k_norm_reshape || !k_norm) {
            return std::nullopt;
        }
        supernode.k_norm_reshape = *k_norm_reshape;
        supernode.k_norm = *k_norm;
    } else if (find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_NORM) != nullptr) {
        return std::nullopt;
    }

    return supernode;
}

[[nodiscard]] std::optional<FastAttentionDecodeSupernode>
build_attention_decode_supernode(std::span<const GraphNode> nodes, std::span<const GraphValue> values) noexcept {
    if (uses_annotated_roles(nodes)) {
        return build_attention_decode_supernode_annotated(nodes, values);
    }
    if (nodes.size() < 13) {
        return std::nullopt;
    }

    size_t idx = 0;
    auto next_rms_norm = [&]() -> std::optional<FastRmsNormOp> {
        if (idx >= nodes.size() || !is_rms_norm_op(nodes[idx].signature.op_id)) {
            return std::nullopt;
        }
        const auto op = build_rms_norm_op(nodes[idx]);
        if (op) {
            idx++;
        }
        return op;
    };
    auto next_matmul = [&]() -> std::optional<FastMatmulOp> {
        if (idx >= nodes.size() || nodes[idx].signature.op_id != MARMOT_OP_MATMUL) {
            return std::nullopt;
        }
        const auto op = build_matmul_op(nodes[idx]);
        if (op) {
            idx++;
        }
        return op;
    };
    auto next_reshape = [&]() -> std::optional<FastReshapeOp> {
        if (idx >= nodes.size() || nodes[idx].signature.op_id != MARMOT_OP_RESHAPE) {
            return std::nullopt;
        }
        const auto op = build_reshape_op(nodes[idx], values);
        if (op) {
            idx++;
        }
        return op;
    };
    auto next_rope = [&]() -> std::optional<FastRopeOp> {
        if (idx >= nodes.size() || nodes[idx].signature.op_id != MARMOT_OP_ROPE) {
            return std::nullopt;
        }
        const auto op = build_rope_op(nodes[idx]);
        if (op) {
            idx++;
        }
        return op;
    };
    auto next_paged_attention = [&]() -> std::optional<FastPagedAttentionOp> {
        if (idx >= nodes.size() || nodes[idx].signature.op_id != MARMOT_OP_PAGED_ATTENTION) {
            return std::nullopt;
        }
        const auto op = build_paged_attention_op(nodes[idx]);
        if (op) {
            idx++;
        }
        return op;
    };
    auto next_add = [&]() -> std::optional<FastBinaryOp> {
        if (idx >= nodes.size() || nodes[idx].signature.op_id != MARMOT_OP_ADD) {
            return std::nullopt;
        }
        const auto op = build_binary_op(nodes[idx]);
        if (op) {
            idx++;
        }
        return op;
    };

    FastAttentionDecodeSupernode supernode{};
    auto attn_norm = next_rms_norm();
    auto q_proj = next_matmul();
    auto k_proj = next_matmul();
    auto v_proj = next_matmul();
    if (!attn_norm || !q_proj || !k_proj || !v_proj) {
        return std::nullopt;
    }
    supernode.attn_norm = *attn_norm;
    supernode.q_proj = *q_proj;
    supernode.k_proj = *k_proj;
    supernode.v_proj = *v_proj;

    if (idx + 1 < nodes.size() && nodes[idx].signature.op_id == MARMOT_OP_RESHAPE &&
        is_rms_norm_op(nodes[idx + 1].signature.op_id)) {
        auto q_norm_reshape = next_reshape();
        auto q_norm = next_rms_norm();
        if (!q_norm_reshape || !q_norm) {
            return std::nullopt;
        }
        supernode.q_norm_reshape = *q_norm_reshape;
        supernode.q_norm = *q_norm;
    }

    if (idx + 1 < nodes.size() && nodes[idx].signature.op_id == MARMOT_OP_RESHAPE &&
        is_rms_norm_op(nodes[idx + 1].signature.op_id)) {
        auto k_norm_reshape = next_reshape();
        auto k_norm = next_rms_norm();
        if (!k_norm_reshape || !k_norm) {
            return std::nullopt;
        }
        supernode.k_norm_reshape = *k_norm_reshape;
        supernode.k_norm = *k_norm;
    }

    auto q_heads = next_reshape();
    auto k_heads = next_reshape();
    auto v_heads = next_reshape();
    auto q_rope = next_rope();
    auto k_rope = next_rope();
    auto q_tokens = next_reshape();
    auto k_tokens = next_reshape();
    auto v_tokens = next_reshape();
    auto paged_attention = next_paged_attention();
    auto attn_flat = next_reshape();
    auto out_proj = next_matmul();
    auto residual_add = next_add();
    if (!q_heads || !k_heads || !v_heads || !q_rope || !k_rope || !q_tokens || !k_tokens || !v_tokens ||
        !paged_attention || !attn_flat || !out_proj || !residual_add) {
        return std::nullopt;
    }

    if (idx != nodes.size()) {
        return std::nullopt;
    }

    supernode.q_heads = *q_heads;
    supernode.k_heads = *k_heads;
    supernode.v_heads = *v_heads;
    supernode.q_rope = *q_rope;
    supernode.k_rope = *k_rope;
    supernode.q_tokens = *q_tokens;
    supernode.k_tokens = *k_tokens;
    supernode.v_tokens = *v_tokens;
    supernode.paged_attention = *paged_attention;
    supernode.attn_flat = *attn_flat;
    supernode.out_proj = *out_proj;
    supernode.residual_add = *residual_add;
    return supernode;
}

[[nodiscard]] StageMatch match_attention_stage(std::span<const GraphNode> nodes, uint32_t start) noexcept {
    if (start >= nodes.size() || !is_rms_norm_op(nodes[start].signature.op_id)) {
        return {};
    }

    uint32_t idx = start + 1;
    auto require = [&](marmot_op_id_t op_id) -> bool {
        return idx < nodes.size() && nodes[idx].signature.op_id == op_id;
    };

    if (!require(MARMOT_OP_MATMUL)) {
        return {};
    }
    idx++;
    if (!require(MARMOT_OP_MATMUL)) {
        return {};
    }
    idx++;
    if (!require(MARMOT_OP_MATMUL)) {
        return {};
    }
    idx++;

    for (int i = 0; i < 2; ++i) {
        if (idx + 1 < nodes.size() && is_reshape_op(nodes[idx].signature.op_id) &&
            is_rms_norm_op(nodes[idx + 1].signature.op_id)) {
            idx += 2;
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (!require(MARMOT_OP_RESHAPE)) {
            return {};
        }
        idx++;
    }
    for (int i = 0; i < 2; ++i) {
        if (!require(MARMOT_OP_ROPE)) {
            return {};
        }
        idx++;
    }
    for (int i = 0; i < 3; ++i) {
        if (!require(MARMOT_OP_RESHAPE)) {
            return {};
        }
        idx++;
    }
    if (!require(MARMOT_OP_PAGED_ATTENTION)) {
        return {};
    }
    idx++;
    if (!require(MARMOT_OP_RESHAPE)) {
        return {};
    }
    idx++;
    if (!require(MARMOT_OP_MATMUL)) {
        return {};
    }
    idx++;
    if (!require(MARMOT_OP_ADD)) {
        return {};
    }
    idx++;

    const uint32_t node_count = idx - start;
    std::array<uint8_t, 8> offsets{};
    const uint32_t lowered_count = std::min<uint32_t>(node_count, offsets.size());
    for (uint32_t i = 0; i < lowered_count; ++i) {
        offsets[i] = static_cast<uint8_t>(i);
    }
    return StageMatch{
        .kind = FastStageKind::Attention,
        .lowering = FastStageLowering::AttentionDecodePaged,
        .node_count = node_count,
        .lowered_node_offsets = offsets,
        .lowered_node_count = static_cast<uint8_t>(lowered_count),
    };
}

[[nodiscard]] std::optional<FastDenseFfnGeluSupernode>
build_dense_ffn_gelu_supernode_annotated(std::span<const GraphNode> nodes) noexcept {
    const GraphNode *rms_norm_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_NORM);
    const GraphNode *up_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_UP_PROJ);
    const GraphNode *gelu_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_GELU);
    const GraphNode *down_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_DOWN_PROJ);
    const GraphNode *residual_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_RESIDUAL);
    if (rms_norm_node == nullptr || up_proj_node == nullptr || gelu_node == nullptr || down_proj_node == nullptr ||
        residual_node == nullptr) {
        return std::nullopt;
    }
    const auto rms_norm = build_rms_norm_op(*rms_norm_node);
    const auto up_proj = build_matmul_op(*up_proj_node);
    const auto gelu = build_unary_op(*gelu_node);
    const auto down_proj = build_matmul_op(*down_proj_node);
    const auto residual_add = build_binary_op(*residual_node);
    if (!rms_norm || !up_proj || !gelu || !down_proj || !residual_add) {
        return std::nullopt;
    }
    return FastDenseFfnGeluSupernode{
        .rms_norm = *rms_norm,
        .up_proj = *up_proj,
        .gelu = *gelu,
        .down_proj = *down_proj,
        .residual_add = *residual_add,
    };
}

[[nodiscard]] std::optional<FastDenseFfnGatedSupernode>
build_dense_ffn_gated_supernode_annotated(std::span<const GraphNode> nodes) noexcept {
    const GraphNode *rms_norm_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_NORM);
    const GraphNode *gate_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_GATE_PROJ);
    const GraphNode *up_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_UP_PROJ);
    const GraphNode *glu_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_GLU);
    const GraphNode *down_proj_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_DOWN_PROJ);
    const GraphNode *residual_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_RESIDUAL);
    if (rms_norm_node == nullptr || gate_proj_node == nullptr || up_proj_node == nullptr || glu_node == nullptr ||
        down_proj_node == nullptr || residual_node == nullptr) {
        return std::nullopt;
    }
    const auto rms_norm = build_rms_norm_op(*rms_norm_node);
    const auto gate_proj = build_matmul_op(*gate_proj_node);
    const auto up_proj = build_matmul_op(*up_proj_node);
    const auto glu = build_binary_op(*glu_node);
    const auto down_proj = build_matmul_op(*down_proj_node);
    const auto residual_add = build_binary_op(*residual_node);
    if (!rms_norm || !gate_proj || !up_proj || !glu || !down_proj || !residual_add) {
        return std::nullopt;
    }
    return FastDenseFfnGatedSupernode{
        .rms_norm = *rms_norm,
        .gate_proj = *gate_proj,
        .up_proj = *up_proj,
        .glu = *glu,
        .down_proj = *down_proj,
        .residual_add = *residual_add,
    };
}

[[nodiscard]] std::optional<FastMoeFfnBasicSupernode>
build_moe_ffn_supernode_annotated(std::span<const GraphNode> nodes, std::span<const GraphValue> values) noexcept {
    const GraphNode *rms_norm_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_NORM);
    const GraphNode *router_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_ROUTER);
    const GraphNode *topk_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_TOPK);
    const GraphNode *moe_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_MOE_EXPERTS);
    const GraphNode *residual_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_RESIDUAL);
    if (rms_norm_node == nullptr || router_node == nullptr || topk_node == nullptr || moe_node == nullptr ||
        residual_node == nullptr) {
        return std::nullopt;
    }
    const auto rms_norm = build_rms_norm_op(*rms_norm_node);
    const auto router = build_matmul_op(*router_node);
    const auto topk = build_topk_op(*topk_node, values);
    const auto moe = build_moe_op(*moe_node);
    const auto residual_add = build_binary_op(*residual_node);
    if (!rms_norm || !router || !topk || !moe || !residual_add) {
        return std::nullopt;
    }

    FastMoeFfnBasicSupernode supernode{
        .rms_norm = *rms_norm,
        .router = *router,
        .topk = *topk,
        .moe = *moe,
        .residual_add = *residual_add,
    };

    if (const GraphNode *softmax_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_ROUTER_PROBS)) {
        const auto softmax = build_softmax_op(*softmax_node);
        if (!softmax) {
            return std::nullopt;
        }
        supernode.topk_first = false;
        supernode.softmax = *softmax;
        return supernode;
    }
    if (const GraphNode *softmax_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_ROUTER_WEIGHTS)) {
        const auto softmax = build_softmax_op(*softmax_node);
        if (!softmax) {
            return std::nullopt;
        }
        supernode.topk_first = true;
        supernode.softmax = *softmax;
        return supernode;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<FastLogitsSupernode>
build_logits_supernode_annotated(std::span<const GraphNode> nodes) noexcept {
    FastLogitsSupernode supernode{};
    if (const GraphNode *final_norm_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_LOGITS_NORM)) {
        const auto final_norm = build_rms_norm_op(*final_norm_node);
        if (!final_norm) {
            return std::nullopt;
        }
        supernode.final_norm = *final_norm;
    }
    if (const GraphNode *gather_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_LOGITS_GATHER)) {
        const auto gather = build_gather_rows_op(*gather_node);
        if (!gather) {
            return std::nullopt;
        }
        supernode.gather = *gather;
    }
    const GraphNode *projection_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_LOGITS_PROJECTION);
    if (projection_node == nullptr) {
        return std::nullopt;
    }
    if (projection_node->signature.op_id == MARMOT_OP_MATMUL) {
        const auto matmul = build_matmul_op(*projection_node);
        if (!matmul) {
            return std::nullopt;
        }
        supernode.matmul = *matmul;
        return supernode;
    }
    if (projection_node->signature.op_id == MARMOT_OP_VEC_DOT) {
        const auto vec_dot = build_vec_dot_op(*projection_node);
        if (!vec_dot) {
            return std::nullopt;
        }
        supernode.vec_dot = *vec_dot;
        return supernode;
    }
    return std::nullopt;
}

[[nodiscard]] FastStagePayload build_stage_payload(
    std::span<const GraphNode> nodes, std::span<const GraphValue> values, FastStageLowering lowering
) noexcept {
    switch (lowering) {
    case FastStageLowering::AttentionDecodePaged: {
        const auto supernode = build_attention_decode_supernode(nodes, values);
        if (!supernode) {
            return {};
        }
        return *supernode;
    }
    case FastStageLowering::DenseFfnGelu: {
        if (uses_annotated_roles(nodes)) {
            if (const auto supernode = build_dense_ffn_gelu_supernode_annotated(nodes)) {
                return *supernode;
            }
            return {};
        }
        if (nodes.size() != 5) {
            return {};
        }
        const auto rms_norm = build_rms_norm_op(nodes[0]);
        const auto up_proj = build_matmul_op(nodes[1]);
        const auto gelu = build_unary_op(nodes[2]);
        const auto down_proj = build_matmul_op(nodes[3]);
        const auto residual_add = build_binary_op(nodes[4]);
        if (!rms_norm || !up_proj || !gelu || !down_proj || !residual_add) {
            return {};
        }
        return FastDenseFfnGeluSupernode{
            .rms_norm = *rms_norm,
            .up_proj = *up_proj,
            .gelu = *gelu,
            .down_proj = *down_proj,
            .residual_add = *residual_add,
        };
    }
    case FastStageLowering::DenseFfnGated: {
        if (uses_annotated_roles(nodes)) {
            if (const auto supernode = build_dense_ffn_gated_supernode_annotated(nodes)) {
                return *supernode;
            }
            return {};
        }
        if (nodes.size() != 6) {
            return {};
        }
        const auto rms_norm = build_rms_norm_op(nodes[0]);
        const auto gate_proj = build_matmul_op(nodes[1]);
        const auto up_proj = build_matmul_op(nodes[2]);
        const auto glu = build_binary_op(nodes[3]);
        const auto down_proj = build_matmul_op(nodes[4]);
        const auto residual_add = build_binary_op(nodes[5]);
        if (!rms_norm || !gate_proj || !up_proj || !glu || !down_proj || !residual_add) {
            return {};
        }
        return FastDenseFfnGatedSupernode{
            .rms_norm = *rms_norm,
            .gate_proj = *gate_proj,
            .up_proj = *up_proj,
            .glu = *glu,
            .down_proj = *down_proj,
            .residual_add = *residual_add,
        };
    }
    case FastStageLowering::MoeFfnBasic: {
        if (uses_annotated_roles(nodes)) {
            if (const auto supernode = build_moe_ffn_supernode_annotated(nodes, values)) {
                return *supernode;
            }
            return {};
        }
        if (nodes.size() != 6) {
            return {};
        }
        const auto rms_norm = build_rms_norm_op(nodes[0]);
        const auto router = build_matmul_op(nodes[1]);
        const auto moe = build_moe_op(nodes[4]);
        const auto residual_add = build_binary_op(nodes[5]);
        if (!rms_norm || !router || !moe || !residual_add) {
            return {};
        }

        FastMoeFfnBasicSupernode supernode{
            .rms_norm = *rms_norm,
            .router = *router,
            .moe = *moe,
            .residual_add = *residual_add,
        };

        if (nodes[2].signature.op_id == MARMOT_OP_TOPK && nodes[3].signature.op_id == MARMOT_OP_SOFTMAX) {
            const auto topk = build_topk_op(nodes[2], values);
            const auto softmax = build_softmax_op(nodes[3]);
            if (!topk || !softmax) {
                return {};
            }
            supernode.topk_first = true;
            supernode.topk = *topk;
            supernode.softmax = *softmax;
            return supernode;
        }

        if (nodes[2].signature.op_id == MARMOT_OP_SOFTMAX && nodes[3].signature.op_id == MARMOT_OP_TOPK) {
            const auto softmax = build_softmax_op(nodes[2]);
            const auto topk = build_topk_op(nodes[3], values);
            if (!softmax || !topk) {
                return {};
            }
            supernode.topk_first = false;
            supernode.softmax = *softmax;
            supernode.topk = *topk;
            return supernode;
        }
        return {};
    }
    case FastStageLowering::LogitsMatmul: {
        if (uses_annotated_roles(nodes)) {
            if (const auto supernode = build_logits_supernode_annotated(nodes)) {
                return *supernode;
            }
            return {};
        }
        if (nodes.size() != 1) {
            return {};
        }
        const auto matmul = build_matmul_op(nodes[0]);
        if (!matmul) {
            return {};
        }
        return FastLogitsSupernode{
            .matmul = *matmul,
        };
    }
    case FastStageLowering::LogitsVecDot: {
        if (uses_annotated_roles(nodes)) {
            if (const auto supernode = build_logits_supernode_annotated(nodes)) {
                return *supernode;
            }
            return {};
        }
        if (nodes.size() != 1) {
            return {};
        }
        const auto vec_dot = build_vec_dot_op(nodes[0]);
        if (!vec_dot) {
            return {};
        }
        return FastLogitsSupernode{
            .vec_dot = *vec_dot,
        };
    }
    case FastStageLowering::LogitsGatherMatmul: {
        if (uses_annotated_roles(nodes)) {
            if (const auto supernode = build_logits_supernode_annotated(nodes)) {
                return *supernode;
            }
            return {};
        }
        if (nodes.size() != 2) {
            return {};
        }
        const auto gather = build_gather_rows_op(nodes[0]);
        const auto matmul = build_matmul_op(nodes[1]);
        if (!gather || !matmul) {
            return {};
        }
        return FastLogitsSupernode{
            .gather = *gather,
            .matmul = *matmul,
        };
    }
    case FastStageLowering::LogitsGatherVecDot: {
        if (uses_annotated_roles(nodes)) {
            if (const auto supernode = build_logits_supernode_annotated(nodes)) {
                return *supernode;
            }
            return {};
        }
        if (nodes.size() != 2) {
            return {};
        }
        const auto gather = build_gather_rows_op(nodes[0]);
        const auto vec_dot = build_vec_dot_op(nodes[1]);
        if (!gather || !vec_dot) {
            return {};
        }
        return FastLogitsSupernode{
            .gather = *gather,
            .vec_dot = *vec_dot,
        };
    }
    case FastStageLowering::None:
        return {};
    }

    return {};
}

[[nodiscard]] FastStageLowering infer_dense_ffn_lowering(std::span<const GraphNode> nodes) noexcept {
    if (uses_annotated_roles(nodes)) {
        const bool has_norm = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_NORM) != nullptr;
        const bool has_up = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_UP_PROJ) != nullptr;
        const bool has_down = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_DOWN_PROJ) != nullptr;
        const bool has_residual = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_RESIDUAL) != nullptr;
        if (has_norm && has_up && has_down && has_residual &&
            find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_GELU) != nullptr) {
            return FastStageLowering::DenseFfnGelu;
        }
        if (has_norm && has_up && has_down && has_residual &&
            find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_GATE_PROJ) != nullptr &&
            find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_GLU) != nullptr) {
            return FastStageLowering::DenseFfnGated;
        }
        return FastStageLowering::None;
    }
    if (nodes.size() == 5 && is_rms_norm_op(nodes[0].signature.op_id) && nodes[1].signature.op_id == MARMOT_OP_MATMUL &&
        nodes[2].signature.op_id == MARMOT_OP_GELU && nodes[3].signature.op_id == MARMOT_OP_MATMUL &&
        is_residual_add_op(nodes[4].signature.op_id)) {
        return FastStageLowering::DenseFfnGelu;
    }
    if (nodes.size() == 6 && is_rms_norm_op(nodes[0].signature.op_id) && nodes[1].signature.op_id == MARMOT_OP_MATMUL &&
        nodes[2].signature.op_id == MARMOT_OP_MATMUL &&
        (nodes[3].signature.op_id == MARMOT_OP_SWIGLU || nodes[3].signature.op_id == MARMOT_OP_GEGLU) &&
        nodes[4].signature.op_id == MARMOT_OP_MATMUL && is_residual_add_op(nodes[5].signature.op_id)) {
        return FastStageLowering::DenseFfnGated;
    }
    return FastStageLowering::None;
}

[[nodiscard]] FastStageLowering infer_moe_ffn_lowering(std::span<const GraphNode> nodes) noexcept {
    if (uses_annotated_roles(nodes)) {
        const bool has_norm = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_NORM) != nullptr;
        const bool has_router = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_ROUTER) != nullptr;
        const bool has_topk = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_TOPK) != nullptr;
        const bool has_moe = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_MOE_EXPERTS) != nullptr;
        const bool has_residual = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_RESIDUAL) != nullptr;
        const bool has_softmax = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_ROUTER_PROBS) != nullptr ||
            find_role_node(nodes, MARMOT_FAST_NODE_ROLE_FFN_ROUTER_WEIGHTS) != nullptr;
        if (has_norm && has_router && has_topk && has_moe && has_residual && has_softmax) {
            return FastStageLowering::MoeFfnBasic;
        }
        return FastStageLowering::None;
    }
    if (nodes.size() != 6 || !is_rms_norm_op(nodes[0].signature.op_id) ||
        nodes[1].signature.op_id != MARMOT_OP_MATMUL || nodes[4].signature.op_id != MARMOT_OP_MOE_EXPERTS ||
        !is_residual_add_op(nodes[5].signature.op_id)) {
        return FastStageLowering::None;
    }
    const marmot_op_id_t op2 = nodes[2].signature.op_id;
    const marmot_op_id_t op3 = nodes[3].signature.op_id;
    if ((op2 == MARMOT_OP_TOPK && op3 == MARMOT_OP_SOFTMAX) || (op2 == MARMOT_OP_SOFTMAX && op3 == MARMOT_OP_TOPK)) {
        return FastStageLowering::MoeFfnBasic;
    }
    return FastStageLowering::None;
}

[[nodiscard]] StageMatch match_ffn_stage(std::span<const GraphNode> nodes, uint32_t start) noexcept {
    if (start >= nodes.size() || !is_rms_norm_op(nodes[start].signature.op_id)) {
        return {};
    }

    bool saw_moe = false;
    bool saw_ffn_math = false;
    for (uint32_t node_index = start + 1; node_index < nodes.size(); ++node_index) {
        const marmot_op_id_t op_id = nodes[node_index].signature.op_id;
        if (is_attention_op(op_id) || op_id == MARMOT_OP_VEC_DOT || op_id == MARMOT_OP_GATHER_ROWS ||
            is_rms_norm_op(op_id)) {
            return {};
        }
        if (is_moe_op(op_id)) {
            saw_moe = true;
            saw_ffn_math = true;
        } else if (is_ffn_compute_op(op_id)) {
            saw_ffn_math = true;
        }
        if (is_residual_add_op(op_id)) {
            if (!saw_ffn_math) {
                return {};
            }
            const uint32_t node_count = node_index - start + 1;
            const auto stage_nodes = nodes.subspan(start, node_count);
            const FastStageKind kind = saw_moe ? FastStageKind::MoeFfn : FastStageKind::DenseFfn;
            const FastStageLowering lowering =
                saw_moe ? infer_moe_ffn_lowering(stage_nodes) : infer_dense_ffn_lowering(stage_nodes);
            const std::array<uint8_t, 8> default_offsets{0, 1, 2, 3, 4, 5, 6, 7};
            return StageMatch{
                .kind = kind,
                .lowering = lowering,
                .node_count = node_count,
                .lowered_node_offsets = default_offsets,
                .lowered_node_count = static_cast<uint8_t>(std::min(node_count, (uint32_t)default_offsets.size())),
            };
        }
    }
    return {};
}

[[nodiscard]] FastStageLowering infer_attention_lowering(std::span<const GraphNode> nodes) noexcept {
    if (uses_annotated_roles(nodes)) {
        const bool has_norm = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_NORM) != nullptr;
        const bool has_q = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_PROJ) != nullptr;
        const bool has_k = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_PROJ) != nullptr;
        const bool has_v = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_V_PROJ) != nullptr;
        const bool has_q_heads = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_HEADS) != nullptr;
        const bool has_k_heads = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_HEADS) != nullptr;
        const bool has_v_heads = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_V_HEADS) != nullptr;
        const bool has_q_rope = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_ROPE) != nullptr;
        const bool has_k_rope = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_ROPE) != nullptr;
        const bool has_q_tokens = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_Q_TOKENS) != nullptr;
        const bool has_k_tokens = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_K_TOKENS) != nullptr;
        const bool has_v_tokens = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_V_TOKENS) != nullptr;
        const bool has_paged = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_PAGED) != nullptr;
        const bool has_out_reshape = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_OUT_RESHAPE) != nullptr;
        const bool has_out_proj = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_OUT_PROJ) != nullptr;
        const bool has_residual = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_ATTN_RESIDUAL) != nullptr;
        if (has_norm && has_q && has_k && has_v && has_q_heads && has_k_heads && has_v_heads && has_q_rope &&
            has_k_rope && has_q_tokens && has_k_tokens && has_v_tokens && has_paged && has_out_reshape &&
            has_out_proj && has_residual) {
            return FastStageLowering::AttentionDecodePaged;
        }
        return FastStageLowering::None;
    }
    return FastStageLowering::None;
}

[[nodiscard]] FastStageLowering infer_logits_lowering(std::span<const GraphNode> nodes) noexcept {
    if (uses_annotated_roles(nodes)) {
        const GraphNode *projection_node = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_LOGITS_PROJECTION);
        if (projection_node == nullptr) {
            return FastStageLowering::None;
        }
        const bool has_gather = find_role_node(nodes, MARMOT_FAST_NODE_ROLE_LOGITS_GATHER) != nullptr;
        if (projection_node->signature.op_id == MARMOT_OP_MATMUL) {
            return has_gather ? FastStageLowering::LogitsGatherMatmul : FastStageLowering::LogitsMatmul;
        }
        if (projection_node->signature.op_id == MARMOT_OP_VEC_DOT) {
            return has_gather ? FastStageLowering::LogitsGatherVecDot : FastStageLowering::LogitsVecDot;
        }
        return FastStageLowering::None;
    }
    return FastStageLowering::None;
}

[[nodiscard]] StageMatch match_annotated_stage(std::span<const GraphNode> nodes, uint32_t start) noexcept {
    if (start >= nodes.size() || !has_fast_stage_hint(nodes[start]) || nodes[start].fast_block_id == UINT32_MAX) {
        return {};
    }
    const marmot_fast_stage_hint_t stage_hint = nodes[start].fast_stage_hint;
    const uint32_t block_id = nodes[start].fast_block_id;
    uint32_t idx = start;
    while (idx < nodes.size() && nodes[idx].fast_stage_hint == stage_hint && nodes[idx].fast_block_id == block_id) {
        idx++;
    }
    const uint32_t node_count = idx - start;
    if (node_count == 0) {
        return {};
    }

    const auto stage_nodes = nodes.subspan(start, node_count);
    const FastStageKind kind = fast_stage_kind_from_hint(stage_hint);
    FastStageLowering lowering = FastStageLowering::None;
    switch (kind) {
    case FastStageKind::Attention:
        lowering = infer_attention_lowering(stage_nodes);
        break;
    case FastStageKind::DenseFfn:
        lowering = infer_dense_ffn_lowering(stage_nodes);
        break;
    case FastStageKind::MoeFfn:
        lowering = infer_moe_ffn_lowering(stage_nodes);
        break;
    case FastStageKind::LogitsTail:
        lowering = infer_logits_lowering(stage_nodes);
        break;
    case FastStageKind::GenericFallback:
        break;
    }

    const std::array<uint8_t, 8> default_offsets{0, 1, 2, 3, 4, 5, 6, 7};
    return StageMatch{
        .kind = kind,
        .lowering = lowering,
        .node_count = node_count,
        .lowered_node_offsets = default_offsets,
        .lowered_node_count = static_cast<uint8_t>(std::min(node_count, (uint32_t)default_offsets.size())),
    };
}

[[nodiscard]] StageMatch match_logits_tail_stage(std::span<const GraphNode> nodes, uint32_t start) noexcept {
    if (start >= nodes.size()) {
        return {};
    }
    const marmot_op_id_t first_op = nodes[start].signature.op_id;
    if (!is_logits_tail_op(first_op)) {
        return {};
    }

    bool saw_projection = first_op == MARMOT_OP_VEC_DOT || first_op == MARMOT_OP_MATMUL;
    for (uint32_t node_index = start; node_index < nodes.size(); ++node_index) {
        const marmot_op_id_t op_id = nodes[node_index].signature.op_id;
        if (!is_logits_tail_op(op_id)) {
            return {};
        }
        saw_projection = saw_projection || op_id == MARMOT_OP_VEC_DOT || op_id == MARMOT_OP_MATMUL;
    }
    if (!saw_projection) {
        return {};
    }
    const uint32_t node_count = static_cast<uint32_t>(nodes.size() - start);
    const auto stage_nodes = nodes.subspan(start, node_count);
    if (stage_nodes.size() == 1 && stage_nodes[0].signature.op_id == MARMOT_OP_MATMUL) {
        constexpr uint8_t offsets[] = {0};
        return make_stage_match(FastStageKind::LogitsTail, FastStageLowering::LogitsMatmul, node_count, offsets);
    }
    if (stage_nodes.size() == 1 && stage_nodes[0].signature.op_id == MARMOT_OP_VEC_DOT) {
        constexpr uint8_t offsets[] = {0};
        return make_stage_match(FastStageKind::LogitsTail, FastStageLowering::LogitsVecDot, node_count, offsets);
    }
    if (stage_nodes.size() == 2 && stage_nodes[0].signature.op_id == MARMOT_OP_GATHER_ROWS &&
        stage_nodes[1].signature.op_id == MARMOT_OP_MATMUL) {
        constexpr uint8_t offsets[] = {0, 1};
        return make_stage_match(FastStageKind::LogitsTail, FastStageLowering::LogitsGatherMatmul, node_count, offsets);
    }
    if (stage_nodes.size() == 2 && stage_nodes[0].signature.op_id == MARMOT_OP_GATHER_ROWS &&
        stage_nodes[1].signature.op_id == MARMOT_OP_VEC_DOT) {
        constexpr uint8_t offsets[] = {0, 1};
        return make_stage_match(FastStageKind::LogitsTail, FastStageLowering::LogitsGatherVecDot, node_count, offsets);
    }
    return StageMatch{
        .kind = FastStageKind::LogitsTail,
        .node_count = node_count,
    };
}

} // namespace

size_t FastPlan::fast_stage_count() const noexcept {
    size_t count = 0;
    for (const Stage &stage : stages_) {
        if (stage.supports_fast_exec) {
            count++;
        }
    }
    return count;
}

void FastPlanCompiler::push_stage(
    FastPlan &plan, FastStageKind kind, FastStageLowering lowering, std::span<const uint8_t> lowered_node_offsets,
    uint32_t first_node, uint32_t node_count, bool supports_fast_exec, bool is_boundary, FastStagePayload payload
) {
    if (node_count == 0) {
        return;
    }
    FastPlan::Stage stage{
        .kind = kind,
        .lowering = lowering,
        .first_node = first_node,
        .node_count = node_count,
        .supports_fast_exec = supports_fast_exec,
        .is_boundary = is_boundary,
        .label = std::string(
            lowering != FastStageLowering::None ? fast_stage_lowering_name(lowering) : fast_stage_kind_name(kind)
        ),
        .payload = std::move(payload),
    };
    const size_t lowered_count = std::min(stage.lowered_node_offsets.size(), lowered_node_offsets.size());
    for (size_t i = 0; i < lowered_count; ++i) {
        stage.lowered_node_offsets[i] = lowered_node_offsets[i];
    }
    stage.lowered_node_count = static_cast<uint8_t>(lowered_count);
    plan.stages_.push_back(stage);
    const uint32_t stage_index = static_cast<uint32_t>(plan.stages_.size() - 1);
    for (uint32_t i = 0; i < node_count; ++i) {
        const uint32_t node_index = first_node + i;
        if (node_index < plan.node_to_stage_.size()) {
            plan.node_to_stage_[node_index] = stage_index;
        }
    }
}

std::expected<FastPlan, marmot_error_t>
FastPlanCompiler::compile(const marmot_graph_t *graph, const FastPlanBucket &bucket) {
    if (graph == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "compile_fast_plan requires graph");
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const Graph::Impl &impl = *graph->inner.impl_;
    FastPlan plan{};
    plan.backend_ = impl.backend;
    plan.phase_ = infer_phase(bucket);
    plan.bucket_ = bucket;
    plan.node_count_ = impl.nodes.size();
    plan.node_to_stage_.assign(impl.nodes.size(), UINT32_MAX);

    uint32_t fallback_first = 0;
    uint32_t fallback_count = 0;

    for (uint32_t node_index = 0; node_index < impl.nodes.size(); ++node_index) {
        const auto remaining = std::span<const GraphNode>(impl.nodes).subspan(node_index);

        if (const StageMatch annotated = match_annotated_stage(remaining, 0); annotated.node_count != 0) {
            FastPlanCompiler::push_stage(
                plan, FastStageKind::GenericFallback, FastStageLowering::None, std::span<const uint8_t>{},
                fallback_first, fallback_count, false, false, {}
            );
            fallback_count = 0;
            FastStagePayload payload{};
            if (annotated.lowering != FastStageLowering::None) {
                payload = build_stage_payload(
                    std::span<const GraphNode>(impl.nodes).subspan(node_index, annotated.node_count), impl.values,
                    annotated.lowering
                );
            }
            const bool supports_fast_exec =
                annotated.lowering != FastStageLowering::None && !std::holds_alternative<std::monostate>(payload);
            FastPlanCompiler::push_stage(
                plan, annotated.kind, annotated.lowering,
                std::span<const uint8_t>(annotated.lowered_node_offsets.data(), annotated.lowered_node_count),
                node_index, annotated.node_count, supports_fast_exec, true, std::move(payload)
            );
            node_index += annotated.node_count - 1;
            continue;
        }

        if (const StageMatch attention = match_attention_stage(remaining, 0); attention.node_count != 0) {
            FastPlanCompiler::push_stage(
                plan, FastStageKind::GenericFallback, FastStageLowering::None, std::span<const uint8_t>{},
                fallback_first, fallback_count, false, false, {}
            );
            fallback_count = 0;
            FastStagePayload payload = build_stage_payload(
                std::span<const GraphNode>(impl.nodes).subspan(node_index, attention.node_count), impl.values,
                attention.lowering
            );
            const bool supports_fast_exec = !std::holds_alternative<std::monostate>(payload);
            FastPlanCompiler::push_stage(
                plan, attention.kind, attention.lowering,
                std::span<const uint8_t>(attention.lowered_node_offsets.data(), attention.lowered_node_count),
                node_index, attention.node_count, supports_fast_exec, true, std::move(payload)
            );
            node_index += attention.node_count - 1;
            continue;
        }

        if (const StageMatch ffn = match_ffn_stage(remaining, 0); ffn.node_count != 0) {
            FastPlanCompiler::push_stage(
                plan, FastStageKind::GenericFallback, FastStageLowering::None, std::span<const uint8_t>{},
                fallback_first, fallback_count, false, false, {}
            );
            fallback_count = 0;
            FastStagePayload payload{};
            if (ffn.lowering != FastStageLowering::None) {
                payload = build_stage_payload(
                    std::span<const GraphNode>(impl.nodes).subspan(node_index, ffn.node_count), impl.values,
                    ffn.lowering
                );
            }
            const bool supports_fast_exec =
                ffn.lowering != FastStageLowering::None && !std::holds_alternative<std::monostate>(payload);
            FastPlanCompiler::push_stage(
                plan, ffn.kind, ffn.lowering,
                std::span<const uint8_t>(ffn.lowered_node_offsets.data(), ffn.lowered_node_count), node_index,
                ffn.node_count, supports_fast_exec, true, std::move(payload)
            );
            node_index += ffn.node_count - 1;
            continue;
        }

        if (const StageMatch logits = match_logits_tail_stage(remaining, 0); logits.node_count != 0) {
            FastPlanCompiler::push_stage(
                plan, FastStageKind::GenericFallback, FastStageLowering::None, std::span<const uint8_t>{},
                fallback_first, fallback_count, false, false, {}
            );
            fallback_count = 0;
            FastStagePayload payload{};
            if (logits.lowering != FastStageLowering::None) {
                payload = build_stage_payload(
                    std::span<const GraphNode>(impl.nodes).subspan(node_index, logits.node_count), impl.values,
                    logits.lowering
                );
            }
            const bool supports_fast_exec =
                logits.lowering != FastStageLowering::None && !std::holds_alternative<std::monostate>(payload);
            FastPlanCompiler::push_stage(
                plan, logits.kind, logits.lowering,
                std::span<const uint8_t>(logits.lowered_node_offsets.data(), logits.lowered_node_count), node_index,
                logits.node_count, supports_fast_exec, true, std::move(payload)
            );
            break;
        }

        const GraphNode &node = impl.nodes[node_index];
        if (!is_attention_op(node.signature.op_id)) {
            if (fallback_count == 0) {
                fallback_first = node_index;
            }
            fallback_count++;
            continue;
        }

        FastPlanCompiler::push_stage(
            plan, FastStageKind::GenericFallback, FastStageLowering::None, std::span<const uint8_t>{}, fallback_first,
            fallback_count, false, false, {}
        );
        fallback_count = 0;

        constexpr uint8_t attention_offsets[] = {0};
        FastPlanCompiler::push_stage(
            plan, FastStageKind::Attention, FastStageLowering::None, attention_offsets, node_index, 1, true, true, {}
        );
    }

    FastPlanCompiler::push_stage(
        plan, FastStageKind::GenericFallback, FastStageLowering::None, std::span<const uint8_t>{}, fallback_first,
        fallback_count, false, false, {}
    );
    return plan;
}

std::expected<FastPlan, marmot_error_t> compile_fast_plan(const marmot_graph_t *graph, const FastPlanBucket &bucket) {
    return FastPlanCompiler::compile(graph, bucket);
}

const char *fast_plan_phase_name(FastPlanPhase phase) noexcept {
    switch (phase) {
    case FastPlanPhase::Unknown:
        return "unknown";
    case FastPlanPhase::Prefill:
        return "prefill";
    case FastPlanPhase::Decode:
        return "decode";
    case FastPlanPhase::Hybrid:
        return "hybrid";
    }
    return "unknown";
}

const char *fast_stage_kind_name(FastStageKind kind) noexcept {
    switch (kind) {
    case FastStageKind::GenericFallback:
        return "generic_fallback";
    case FastStageKind::Attention:
        return "attention";
    case FastStageKind::DenseFfn:
        return "dense_ffn";
    case FastStageKind::MoeFfn:
        return "moe_ffn";
    case FastStageKind::LogitsTail:
        return "logits_tail";
    }
    return "generic_fallback";
}

const char *fast_stage_lowering_name(FastStageLowering lowering) noexcept {
    switch (lowering) {
    case FastStageLowering::None:
        return "none";
    case FastStageLowering::AttentionDecodePaged:
        return "attention_decode_paged";
    case FastStageLowering::DenseFfnGelu:
        return "dense_ffn_gelu";
    case FastStageLowering::DenseFfnGated:
        return "dense_ffn_gated";
    case FastStageLowering::MoeFfnBasic:
        return "moe_ffn_basic";
    case FastStageLowering::LogitsMatmul:
        return "logits_matmul";
    case FastStageLowering::LogitsVecDot:
        return "logits_vec_dot";
    case FastStageLowering::LogitsGatherMatmul:
        return "logits_gather_matmul";
    case FastStageLowering::LogitsGatherVecDot:
        return "logits_gather_vec_dot";
    }
    return "none";
}

} // namespace marmot::graph
