#pragma once

#include "marmot/error.h"
#include "marmot/tensor.h"

#include <span>

#include "fast_executor.hpp"
#include "fast_plan.hpp"
#include "graph_impl.hpp"

namespace marmot::graph {

class ExecutionSession;
class Executor;
struct GraphNode;

class FastExec {
  public:
    [[nodiscard]] static bool supports(const marmot_context_t *ctx, const FastPlan &plan) noexcept;

    [[nodiscard]] static marmot_error_t execute(
        Graph::Impl &impl, ExecutionSession *session, const FastPlan &plan, const marmot_context_t *ctx,
        std::span<marmot_tensor_t *> bindings, FastExecProfile *profile
    );

  private:
    [[nodiscard]] static marmot_error_t execute_stage(
        Graph::Impl &impl, ExecutionSession *session, Executor &executor, const FastPlan::Stage &stage,
        const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings, FastExecProfileStage *profile_stage
    );

    [[nodiscard]] static marmot_error_t prepare_node_rope(
        ExecutionSession *session, Executor &executor, const GraphNode &node, std::span<marmot_tensor_t *> bindings,
        marmot_tensor_t *&restore_ptr, marmot_value_id_t &restore_id
    );
};

} // namespace marmot::graph
