#pragma once

#include "marmot/graph/graph.h"

#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include "fast_plan.hpp"

namespace marmot::graph {

struct FastExecProfileStage {
    FastStageKind kind{FastStageKind::GenericFallback};
    uint32_t first_node{0};
    uint32_t node_count{0};
    uint32_t executed_ops{0};
    uint64_t duration_ns{0};
};

struct FastExecProfile {
    marmot_backend_type_t backend{MARMOT_BACKEND_CPU};
    FastPlanPhase phase{FastPlanPhase::Unknown};
    uint64_t total_ns{0};
    std::vector<FastExecProfileStage> stages{};
};

class FastExecutor {
  public:
    [[nodiscard]] static marmot_error_t execute(
        marmot_graph_t *graph, const FastPlan *plan, const marmot_context_t *ctx,
        std::span<const marmot_tensor_t *const> inputs, std::span<marmot_tensor_t *const> outputs,
        FastExecProfile *profile
    );

    static void print_profile(const FastExecProfile &profile, FILE *out = stderr);
};

} // namespace marmot::graph
