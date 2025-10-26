#include "marmot/graph/graph.hpp"
#include "marmot/stride_utils.h"

#include "core/dispatch/fusion_detection.h"
#include "core/dispatch/fusion_flags.h"
#include "graph_impl.hpp"
#include "kernel_query.hpp"

namespace marmot::graph {

void Graph::apply_fusion_pass(Impl &impl, marmot_backend_type_t backend) {
    for (size_t i = 0; i + 1 < impl.nodes.size(); ++i) {
        auto &node = impl.nodes[i];
        if (node.outputs.empty())
            continue;
        marmot_value_id_t out_id = node.outputs[0];
        if (out_id >= impl.values.size())
            continue;
        const auto &out_val = impl.values[out_id];
        if (out_val.is_input || out_val.is_constant)
            continue;
        if (out_val.uses.size() != 1)
            continue;
        uint32_t next_idx = out_val.uses[0];
        if (next_idx != i + 1 || next_idx >= impl.nodes.size())
            continue;
        auto &next_node = impl.nodes[next_idx];
        if (next_node.skip)
            continue;

        marmot_op_id_t next_next_op = MARMOT_OP_INVALID;
        bool next_intermediate_temporary = false;
        uint32_t next_next_idx = 0;
        GraphNode *next_next_node = nullptr;
        if (!next_node.outputs.empty()) {
            marmot_value_id_t next_out_id = next_node.outputs[0];
            if (next_out_id < impl.values.size()) {
                const auto &next_out_val = impl.values[next_out_id];
                if (!next_out_val.is_input && !next_out_val.is_constant && next_out_val.uses.size() == 1) {
                    next_next_idx = next_out_val.uses[0];
                    if (next_next_idx == next_idx + 1 && next_next_idx < impl.nodes.size()) {
                        auto &candidate = impl.nodes[next_next_idx];
                        if (!candidate.skip) {
                            next_next_node = &candidate;
                            next_next_op = candidate.signature.op_id;
                            next_intermediate_temporary = true;
                        }
                    }
                }
            }
        }

        marmot_fusion_context_t fusion_ctx = {
            .prev_op = i > 0 ? impl.nodes[i - 1].signature.op_id : MARMOT_OP_INVALID,
            .current_op = node.signature.op_id,
            .next_op = next_node.signature.op_id,
            .next_next_op = next_next_op,
            .intermediate = nullptr,
            .intermediate_is_temporary = true,
            .next_intermediate_is_temporary = next_intermediate_temporary,
            .detected_pattern = MARMOT_FUSION_PATTERN_NONE,
        };

        marmot_op_id_t fused_op = marmot_detect_fused_op_id(&fusion_ctx);
        if (fused_op == MARMOT_OP_INVALID)
            continue;
        const bool matmul_bias_fused =
            (fused_op == MARMOT_OP_MATMUL_BIAS || fused_op == MARMOT_OP_MATMUL_BIAS_RELU ||
             fused_op == MARMOT_OP_MATMUL_BIAS_GELU || fused_op == MARMOT_OP_MATMUL_BIAS_SILU);
        const bool matmul_bias_activation =
            (fused_op == MARMOT_OP_MATMUL_BIAS_RELU || fused_op == MARMOT_OP_MATMUL_BIAS_GELU ||
             fused_op == MARMOT_OP_MATMUL_BIAS_SILU);
        const bool mul_add_fused = (fused_op == MARMOT_OP_FMA);
        if (matmul_bias_fused) {
            if (node.signature.epilogue_flags != MARMOT_EPILOGUE_NONE) {
                continue;
            }
        }

        auto unfused_current = query_backend_for_node(backend, &node.signature);
        auto unfused_next = query_backend_for_node(backend, &next_node.signature);
        double unfused_cost = unfused_current.estimated_us + unfused_next.estimated_us;
        if (matmul_bias_activation) {
            if (next_next_node == nullptr) {
                continue;
            }
            auto unfused_activation = query_backend_for_node(backend, &next_next_node->signature);
            unfused_cost += unfused_activation.estimated_us;
        }

        marmot_op_signature_t fused_sig = node.signature;
        fused_sig.op_id = fused_op;
        fused_sig.variant_flags = MARMOT_FUSION_NONE;
        if (matmul_bias_fused) {
            fused_sig.epilogue_flags = MARMOT_EPILOGUE_BIAS;
            if (matmul_bias_activation) {
                switch (fused_op) {
                case MARMOT_OP_MATMUL_BIAS_RELU:
                    fused_sig.activation = MARMOT_DEVICE_UNARY_RELU;
                    break;
                case MARMOT_OP_MATMUL_BIAS_GELU:
                    fused_sig.activation = MARMOT_DEVICE_UNARY_GELU;
                    break;
                case MARMOT_OP_MATMUL_BIAS_SILU:
                    fused_sig.activation = MARMOT_DEVICE_UNARY_SILU;
                    break;
                default:
                    fused_sig.activation = MARMOT_DEVICE_UNARY_IDENTITY;
                    break;
                }
            } else {
                fused_sig.activation = MARMOT_DEVICE_UNARY_IDENTITY;
            }
        }
        auto fused_result = query_backend_for_node(backend, &fused_sig);

        if (!fused_result.supported || fused_result.estimated_us >= unfused_cost)
            continue;

        if (matmul_bias_fused) {
            if (next_node.inputs.size() != 2)
                continue;
            marmot_value_id_t bias_id = MARMOT_VALUE_ID_INVALID;
            if (next_node.inputs[0] == out_id) {
                bias_id = next_node.inputs[1];
            } else if (next_node.inputs[1] == out_id) {
                bias_id = next_node.inputs[0];
            }
            if (bias_id == MARMOT_VALUE_ID_INVALID)
                continue;
            node.inputs.push_back(bias_id);
        }
        if (mul_add_fused) {
            if (node.inputs.size() != 2 || next_node.inputs.size() != 2)
                continue;
            marmot_value_id_t add_input = MARMOT_VALUE_ID_INVALID;
            if (next_node.inputs[0] == out_id) {
                add_input = next_node.inputs[1];
            } else if (next_node.inputs[1] == out_id) {
                add_input = next_node.inputs[0];
            }
            if (add_input == MARMOT_VALUE_ID_INVALID) {
                continue;
            }
            const auto &add_desc = impl.values[add_input].desc;
            marmot_stride_mode_t add_mode =
                marmot_stride_mode_from_layout(add_desc.ndim, add_desc.shape, add_desc.strides);
            if (add_mode > fused_sig.stride_mode) {
                fused_sig.stride_mode = add_mode;
            }
            node.inputs.push_back(add_input);
        }

        node.signature = fused_sig;
        if (matmul_bias_activation) {
            if (next_next_node == nullptr || next_next_node->outputs.empty()) {
                continue;
            }
            node.outputs = next_next_node->outputs;
            next_next_node->skip = true;
        } else {
            node.outputs = next_node.outputs;
        }
        next_node.skip = true;
    }
}

} // namespace marmot::graph
