#pragma once

#include "marmot/error.h"
#include "marmot/graph/graph_types.h"
#include "marmot/graph/op_signature.h"
#include "marmot/tensor.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace marmot::graph {

class Executor;
class ExecutionSession;
class FastPlanCompiler;
class FastExecutor;
class FastExec;

class Graph {
  public:
    Graph();
    ~Graph();

    Graph(const Graph &) = delete;
    Graph &operator=(const Graph &) = delete;
    Graph(Graph &&) noexcept;
    Graph &operator=(Graph &&) noexcept;

    [[nodiscard]] marmot_backend_type_t backend() const;

    [[nodiscard]] std::expected<marmot_value_id_t, marmot_error_t> add_input(const marmot_graph_tensor_desc_t &desc);

    [[nodiscard]] std::expected<std::vector<marmot_value_id_t>, marmot_error_t> add_op(
        std::string_view op_name, const marmot_op_signature_t *signature, std::span<const marmot_value_id_t> inputs,
        std::span<const marmot_graph_tensor_desc_t> output_descs
    );

    [[nodiscard]] marmot_error_t set_constant(marmot_value_id_t id, marmot_tensor_t *tensor);
    [[nodiscard]] marmot_error_t set_name(marmot_value_id_t id, std::string_view name);
    [[nodiscard]] marmot_error_t
    set_inference_hints(size_t max_seq_len, const marmot_rope_params_t *rope_params, float rms_norm_eps);
    [[nodiscard]] std::expected<marmot_graph_tensor_desc_t, marmot_error_t> get_value_desc(marmot_value_id_t id) const;

    [[nodiscard]] marmot_error_t finalize(marmot_backend_type_t backend);
    [[nodiscard]] marmot_error_t finalize_auto(marmot_backend_type_t *out_backend);
    [[nodiscard]] marmot_error_t
    finalize_auto_with_policy(marmot_routing_policy_t policy, marmot_backend_type_t *out_backend);

    [[nodiscard]] marmot_error_t execute(
        const marmot_context_t *ctx, std::span<const marmot_tensor_t *const> inputs,
        std::span<marmot_tensor_t *const> outputs
    );

    [[nodiscard]] marmot_error_t dump_json(const char *path) const;

    [[nodiscard]] float estimated_total_us() const;
    [[nodiscard]] size_t node_count() const;
    [[nodiscard]] size_t fused_node_count() const;

    void reset_session();

    void set_last_node_view_byte_offset(size_t byte_offset);
    void set_last_node_slice_starts(const size_t *starts, size_t ndim);
    void set_last_node_paged_attention_layer(uint32_t layer_idx);
    [[nodiscard]] marmot_error_t set_last_node_fast_hint_checked(
        marmot_fast_stage_hint_t stage_hint, marmot_fast_node_role_t role, uint32_t block_id
    );
    [[nodiscard]] marmot_error_t set_last_node_moe_params_checked(
        marmot_ffn_type_t ffn_type, float weights_scale, marmot_router_weight_policy_t router_weight_policy
    );
    [[nodiscard]] marmot_error_t set_last_node_moe_params_checked(marmot_ffn_type_t ffn_type, float weights_scale);
    void set_last_node_fast_hint(marmot_fast_stage_hint_t stage_hint, marmot_fast_node_role_t role, uint32_t block_id);
    void set_last_node_moe_params(
        marmot_ffn_type_t ffn_type, float weights_scale, marmot_router_weight_policy_t router_weight_policy
    );
    void set_last_node_moe_params(marmot_ffn_type_t ffn_type, float weights_scale);

  private:
    friend class Executor;
    friend class ExecutionSession;
    friend class FastPlanCompiler;
    friend class FastExecutor;
    friend class FastExec;
    struct Impl;
    [[nodiscard]] static float estimated_total_us(const Impl &impl);
    [[nodiscard]] static marmot_error_t finalize_impl(Impl &impl, marmot_backend_type_t backend, bool emit_errors);
    static void apply_fusion_pass(Impl &impl, marmot_backend_type_t backend);
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<ExecutionSession> session_;
};

} // namespace marmot::graph
