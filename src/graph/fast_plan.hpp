#pragma once

#include "marmot/error.h"
#include "marmot/graph/graph.h"
#include "marmot/types.h"

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "core/bytecode/bytecode.h"

namespace marmot::graph {

enum class FastPlanPhase : uint8_t {
    Unknown = 0,
    Prefill,
    Decode,
    Hybrid,
};

enum class FastStageKind : uint8_t {
    GenericFallback = 0,
    Attention,
    DenseFfn,
    MoeFfn,
    LogitsTail,
};

enum class FastStageLowering : uint8_t {
    None = 0,
    AttentionDecodePaged,
    DenseFfnGelu,
    DenseFfnGated,
    MoeFfnBasic,
    LogitsMatmul,
    LogitsVecDot,
    LogitsGatherMatmul,
    LogitsGatherVecDot,
};

struct FastPlanBucket {
    uint32_t token_count{0};
    uint32_t sample_count{0};
    bool emit_logits{false};
};

struct FastBytecodeOpRef {
    marmot_op_signature_t signature{};
    uint16_t bc_op_index{MARMOT_BC_OP_INVALID};
};

struct FastRmsNormOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t weight{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
};

struct FastMatmulOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t weight{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t bias{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
};

struct FastUnaryOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
};

struct FastBinaryOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input_a{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t input_b{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
};

struct FastSoftmaxOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
    int32_t axis{-1};
};

struct FastTopkOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t values_out{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t indices_out{MARMOT_VALUE_ID_INVALID};
    int32_t axis{-1};
    uint32_t k{0};
};

struct FastMoeExpertsOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t hidden_states{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t gate_exps{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t up_exps{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t down_exps{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t topk_ids{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t topk_weights{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
    marmot_ffn_type_t ffn_type{MARMOT_FFN_COUNT};
    float weights_scale{1.0f};
    marmot_router_weight_policy_t router_weight_policy{MARMOT_ROUTER_WEIGHT_POLICY_COUNT};
};

struct FastGatherRowsOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t indices{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
};

struct FastVecDotOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t weight{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
};

struct FastReshapeOp {
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
    marmot_graph_tensor_desc_t output_desc{};
};

struct FastRopeOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t input{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t positions{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
};

struct FastPagedAttentionOp {
    FastBytecodeOpRef op{};
    marmot_value_id_t token_meta{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t q{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t k{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t v{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t kv_k{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t kv_v{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t block_table{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t kv_k_scale{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t kv_v_scale{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t output{MARMOT_VALUE_ID_INVALID};
    uint32_t layer_idx{0};
};

struct FastDenseFfnGeluSupernode {
    FastRmsNormOp rms_norm{};
    FastMatmulOp up_proj{};
    FastUnaryOp gelu{};
    FastMatmulOp down_proj{};
    FastBinaryOp residual_add{};
};

struct FastDenseFfnGatedSupernode {
    FastRmsNormOp rms_norm{};
    FastMatmulOp gate_proj{};
    FastMatmulOp up_proj{};
    FastBinaryOp glu{};
    FastMatmulOp down_proj{};
    FastBinaryOp residual_add{};
};

struct FastMoeFfnBasicSupernode {
    FastRmsNormOp rms_norm{};
    FastMatmulOp router{};
    bool topk_first{true};
    FastTopkOp topk{};
    FastSoftmaxOp softmax{};
    FastMoeExpertsOp moe{};
    FastBinaryOp residual_add{};
};

struct FastAttentionDecodeSupernode {
    FastRmsNormOp attn_norm{};
    FastMatmulOp q_proj{};
    FastMatmulOp k_proj{};
    FastMatmulOp v_proj{};
    std::optional<FastReshapeOp> q_norm_reshape{};
    std::optional<FastRmsNormOp> q_norm{};
    std::optional<FastReshapeOp> k_norm_reshape{};
    std::optional<FastRmsNormOp> k_norm{};
    FastReshapeOp q_heads{};
    FastReshapeOp k_heads{};
    FastReshapeOp v_heads{};
    FastRopeOp q_rope{};
    FastRopeOp k_rope{};
    FastReshapeOp q_tokens{};
    FastReshapeOp k_tokens{};
    FastReshapeOp v_tokens{};
    FastPagedAttentionOp paged_attention{};
    FastReshapeOp attn_flat{};
    FastMatmulOp out_proj{};
    FastBinaryOp residual_add{};
};

struct FastLogitsSupernode {
    std::optional<FastRmsNormOp> final_norm{};
    std::optional<FastGatherRowsOp> gather{};
    std::optional<FastMatmulOp> matmul{};
    std::optional<FastVecDotOp> vec_dot{};
};

using FastStagePayload = std::variant<
    std::monostate, FastAttentionDecodeSupernode, FastDenseFfnGeluSupernode, FastDenseFfnGatedSupernode,
    FastMoeFfnBasicSupernode, FastLogitsSupernode>;

class FastPlan {
  public:
    struct Stage {
        FastStageKind kind{FastStageKind::GenericFallback};
        FastStageLowering lowering{FastStageLowering::None};
        uint32_t first_node{0};
        uint32_t node_count{0};
        bool supports_fast_exec{false};
        bool is_boundary{false};
        std::array<uint8_t, 8> lowered_node_offsets{};
        uint8_t lowered_node_count{0};
        std::string label{};
        FastStagePayload payload{};
    };

    [[nodiscard]] marmot_backend_type_t backend() const noexcept {
        return backend_;
    }

    [[nodiscard]] FastPlanPhase phase() const noexcept {
        return phase_;
    }

    [[nodiscard]] const FastPlanBucket &bucket() const noexcept {
        return bucket_;
    }

    [[nodiscard]] std::span<const Stage> stages() const noexcept {
        return stages_;
    }

    [[nodiscard]] std::span<const uint32_t> node_to_stage() const noexcept {
        return node_to_stage_;
    }

    [[nodiscard]] size_t node_count() const noexcept {
        return node_count_;
    }

    [[nodiscard]] size_t fast_stage_count() const noexcept;

  private:
    friend class FastPlanCompiler;
    friend std::expected<FastPlan, marmot_error_t>
    compile_fast_plan(const marmot_graph_t *graph, const FastPlanBucket &bucket);

    marmot_backend_type_t backend_{MARMOT_BACKEND_CPU};
    FastPlanPhase phase_{FastPlanPhase::Unknown};
    FastPlanBucket bucket_{};
    std::vector<Stage> stages_{};
    std::vector<uint32_t> node_to_stage_{};
    size_t node_count_{0};
};

[[nodiscard]] std::expected<FastPlan, marmot_error_t>
compile_fast_plan(const marmot_graph_t *graph, const FastPlanBucket &bucket);

[[nodiscard]] const char *fast_plan_phase_name(FastPlanPhase phase) noexcept;
[[nodiscard]] const char *fast_stage_kind_name(FastStageKind kind) noexcept;
[[nodiscard]] const char *fast_stage_lowering_name(FastStageLowering lowering) noexcept;

} // namespace marmot::graph
