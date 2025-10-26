#pragma once

#include "marmot/graph/graph.hpp"
#include "marmot/types.h"

#include <memory>
#include <utility>
#include <vector>

#include "core/bytecode/bytecode.h"
#include "graph_node.hpp"
#include "graph_value.hpp"

namespace marmot::graph {

struct GraphRopeInfo {
    uint32_t node_index{0};
    marmot_value_id_t input_id{MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t positions_id{MARMOT_VALUE_ID_INVALID};
};

struct Graph::Impl {
    std::vector<GraphValue> values;
    std::vector<GraphNode> nodes;
    std::vector<ExecutionCommand> plan;
    std::vector<uint32_t> bc_instr_nodes;
    std::vector<GraphRopeInfo> rope_nodes;
    marmot_bc_program_t program{};
    bool finalized{false};
    marmot_backend_type_t backend{MARMOT_BACKEND_CPU};

    Impl() = default;

    Impl(const Impl &other)
        : values(other.values), nodes(other.nodes), plan(other.plan), bc_instr_nodes(other.bc_instr_nodes),
          rope_nodes(other.rope_nodes), program{}, finalized(other.finalized), backend(other.backend),
          inference(other.inference) {}

    Impl &operator=(const Impl &other) {
        if (this == &other) {
            return *this;
        }
        marmot_bc_program_destroy(&program);
        values = other.values;
        nodes = other.nodes;
        plan = other.plan;
        bc_instr_nodes = other.bc_instr_nodes;
        rope_nodes = other.rope_nodes;
        program = {};
        finalized = other.finalized;
        backend = other.backend;
        inference = other.inference;
        return *this;
    }

    Impl(Impl &&other) noexcept
        : values(std::move(other.values)), nodes(std::move(other.nodes)), plan(std::move(other.plan)),
          bc_instr_nodes(std::move(other.bc_instr_nodes)), rope_nodes(std::move(other.rope_nodes)),
          program(other.program), finalized(other.finalized), backend(other.backend), inference(other.inference) {
        other.program = {};
        other.finalized = false;
    }

    Impl &operator=(Impl &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        marmot_bc_program_destroy(&program);
        values = std::move(other.values);
        nodes = std::move(other.nodes);
        plan = std::move(other.plan);
        bc_instr_nodes = std::move(other.bc_instr_nodes);
        rope_nodes = std::move(other.rope_nodes);
        program = other.program;
        other.program = {};
        finalized = other.finalized;
        backend = other.backend;
        inference = other.inference;
        return *this;
    }

    struct InferenceHints {
        size_t max_seq_len{0};
        float rope_theta{10000.0f};
        marmot_rope_scaling_type_t rope_scaling_type{MARMOT_ROPE_SCALING_NONE};
        marmot_rope_type_t rope_type{MARMOT_ROPE_TYPE_NORM};
        float rope_freq_scale{1.0f};
        float rope_ext_factor{0.0f};
        float rope_attn_factor{1.0f};
        float rope_beta_fast{32.0f};
        float rope_beta_slow{1.0f};
        uint32_t rope_orig_ctx_len{0};
        uint32_t rope_head_dim{0};
        float rms_norm_eps{1e-5f};
    } inference;

    [[nodiscard]] bool is_valid_id(marmot_value_id_t id) const {
        return id < values.size();
    }

    ~Impl() {
        marmot_bc_program_destroy(&program);
    }
};

} // namespace marmot::graph
