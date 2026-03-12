#include "fast_executor.hpp"

#include "marmot/device.h"
#include "marmot/error.h"

#include <memory>

#include "execution_session.hpp"
#include "graph_handle.hpp"

namespace marmot::graph {

marmot_error_t FastExecutor::execute(
    marmot_graph_t *graph, const FastPlan *plan, const marmot_context_t *ctx,
    std::span<const marmot_tensor_t *const> inputs, std::span<marmot_tensor_t *const> outputs, FastExecProfile *profile
) {
    if (graph == nullptr || plan == nullptr || ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (graph->inner.impl_ == nullptr || !graph->inner.impl_->finalized) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    if (ctx->backend_type != graph->inner.impl_->backend) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Context backend mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (plan->backend() != graph->inner.impl_->backend) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast plan backend mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (plan->node_count() != graph->inner.impl_->nodes.size()) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast plan node count mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (graph->inner.session_ == nullptr || !graph->inner.session_->compatible(ctx)) {
        graph->inner.session_ = std::make_unique<ExecutionSession>(*graph->inner.impl_);
        marmot_error_t init_status = graph->inner.session_->initialize(ctx);
        if (init_status != MARMOT_SUCCESS) {
            graph->inner.session_.reset();
            return init_status;
        }
    }

    return graph->inner.session_->execute_with_fast_plan(ctx, inputs, outputs, plan, profile);
}

void FastExecutor::print_profile(const FastExecProfile &profile, FILE *out) {
    if (out == nullptr) {
        return;
    }
    std::fprintf(
        out, "[fast plan] backend=%d phase=%s total=%.3fms stages=%zu\n", (int)profile.backend,
        fast_plan_phase_name(profile.phase), (double)profile.total_ns / 1000000.0, profile.stages.size()
    );
    for (size_t i = 0; i < profile.stages.size(); ++i) {
        const FastExecProfileStage &stage = profile.stages[i];
        std::fprintf(
            out, "[fast plan] stage=%zu kind=%s nodes=%u..%u ops=%u time=%.3fms\n", i, fast_stage_kind_name(stage.kind),
            stage.first_node, stage.node_count == 0 ? stage.first_node : stage.first_node + stage.node_count - 1,
            stage.executed_ops, (double)stage.duration_ns / 1000000.0
        );
    }
}

} // namespace marmot::graph
