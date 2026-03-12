#pragma once

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/tensor.h"

#include <span>

#include "graph_impl.hpp"

namespace marmot::graph {

class ExecutionSession;
class FastExec;
class FastPlan;
struct FastExecProfile;

class Executor {
  public:
    explicit Executor(Graph::Impl &graph_impl, ExecutionSession *session = nullptr);

    [[nodiscard]] marmot_error_t execute(
        const marmot_context_t *ctx, std::span<const marmot_tensor_t *const> inputs,
        std::span<marmot_tensor_t *const> outputs
    );

    [[nodiscard]] marmot_error_t execute_bound(
        const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings, const FastPlan *fast_plan = nullptr,
        FastExecProfile *profile = nullptr
    );

  private:
    friend class FastExec;
    Graph::Impl &impl_;
    ExecutionSession *session_{nullptr};

    [[nodiscard]] marmot_error_t execute_node(
        uint32_t node_index, const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings
    );

    [[nodiscard]] marmot_error_t
    execute_matmul(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_qkv(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_rms_norm(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_quantize(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_convert(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_dequantize(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_embedding(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_rope(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_unary(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_binary(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_softmax(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_topk(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_moe_experts(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_layernorm(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t execute_attention(
        uint32_t node_index, const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings
    );

    [[nodiscard]] marmot_error_t
    execute_paged_attention(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_vec_dot(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_reduction(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_reshape(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_transpose(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_concat(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_slice(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_gather_rows(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_view(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t
    execute_contiguous(const GraphNode &node, const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings);

    [[nodiscard]] marmot_error_t update_qkv_rope_params(std::span<marmot_tensor_t *> bindings);
};

} // namespace marmot::graph
