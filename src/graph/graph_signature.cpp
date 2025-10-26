#include "graph_signature.hpp"

#include "marmot/stride_utils.h"

#include <algorithm>

#include "core/dispatch/signature_infer.h"

namespace marmot::graph {

static marmot_stride_mode_t graph_desc_stride_mode(const marmot_graph_tensor_desc_t &desc) {
    return marmot_stride_mode_from_layout(desc.ndim, desc.shape, desc.strides);
}

static marmot_stride_mode_t merge_stride_modes(marmot_stride_mode_t a, marmot_stride_mode_t b) {
    return a > b ? a : b;
}

static marmot_stride_mode_t infer_stride_mode(const std::vector<GraphValue> &values, const GraphNode &node) {
    marmot_stride_mode_t mode = MARMOT_STRIDE_MODE_CONTIGUOUS;
    for (marmot_value_id_t input_id : node.inputs) {
        mode = merge_stride_modes(mode, graph_desc_stride_mode(values[input_id].desc));
    }
    for (marmot_value_id_t output_id : node.outputs) {
        mode = merge_stride_modes(mode, graph_desc_stride_mode(values[output_id].desc));
    }
    return mode;
}

bool infer_matmul_signature(const std::vector<GraphValue> &values, GraphNode &node) {
    if (node.inputs.size() < 2 || node.outputs.empty())
        return false;

    const auto &input_desc = values[node.inputs[0]].desc;
    const auto &weight_desc = values[node.inputs[1]].desc;
    const auto &output_desc = values[node.outputs[0]].desc;
    return marmot_signature_infer_matmul(&input_desc, &weight_desc, &output_desc, &node.signature);
}

static bool infer_elementwise_signature(const std::vector<GraphValue> &values, GraphNode &node) {
    if (node.inputs.empty() || node.outputs.empty())
        return false;

    const auto &input_desc = values[node.inputs[0]].desc;
    const auto &output_desc = values[node.outputs[0]].desc;
    const marmot_graph_tensor_desc_t *weight_desc = node.inputs.size() >= 2 ? &values[node.inputs[1]].desc : nullptr;
    return marmot_signature_infer_elementwise(&input_desc, weight_desc, &output_desc, &node.signature);
}

static bool infer_paged_attention_signature(const std::vector<GraphValue> &values, GraphNode &node) {
    if (node.inputs.size() < 5 || node.outputs.empty())
        return false;

    const auto &q_desc = values[node.inputs[1]].desc;
    const auto &kv_desc = values[node.inputs[4]].desc;
    const auto &output_desc = values[node.outputs[0]].desc;
    return marmot_signature_infer_paged_attention(&q_desc, &kv_desc, &output_desc, &node.signature);
}

static bool infer_passthrough_signature(const std::vector<GraphValue> &values, GraphNode &node) {
    if (node.inputs.empty() || node.outputs.empty())
        return false;

    const auto &input_desc = values[node.inputs[0]].desc;
    const auto &output_desc = values[node.outputs[0]].desc;
    return marmot_signature_infer_passthrough(&input_desc, &output_desc, &node.signature);
}

bool populate_signature(const std::vector<GraphValue> &values, GraphNode &node) {
    const marmot_op_id_t op_id = node.signature.op_id;
    bool ok = false;

    if (op_id == MARMOT_OP_MATMUL || op_id == MARMOT_OP_LINEAR || op_id == MARMOT_OP_MATMUL_BIAS ||
        op_id == MARMOT_OP_MATMUL_BIAS_RELU || op_id == MARMOT_OP_MATMUL_BIAS_GELU ||
        op_id == MARMOT_OP_MATMUL_BIAS_SILU || op_id == MARMOT_OP_QKV_ROPE || op_id == MARMOT_OP_QKV_SHARED_INPUT ||
        op_id == MARMOT_OP_QKV_PROJECTION) {
        ok = infer_matmul_signature(values, node);
    } else if (op_id == MARMOT_OP_ADD || op_id == MARMOT_OP_ADD_RELU || op_id == MARMOT_OP_ADD_GELU ||
               op_id == MARMOT_OP_ADD_SILU || op_id == MARMOT_OP_SUB || op_id == MARMOT_OP_MUL ||
               op_id == MARMOT_OP_DIV || op_id == MARMOT_OP_MIN || op_id == MARMOT_OP_MAX || op_id == MARMOT_OP_POW ||
               op_id == MARMOT_OP_MOD || op_id == MARMOT_OP_FMA || op_id == MARMOT_OP_WHERE) {
        ok = infer_elementwise_signature(values, node);
    } else if (op_id == MARMOT_OP_SILU || op_id == MARMOT_OP_RELU || op_id == MARMOT_OP_GELU ||
               op_id == MARMOT_OP_GELU_TANH || op_id == MARMOT_OP_SIGMOID || op_id == MARMOT_OP_TANH ||
               op_id == MARMOT_OP_MISH || op_id == MARMOT_OP_ELU || op_id == MARMOT_OP_SELU ||
               op_id == MARMOT_OP_LEAKY_RELU || op_id == MARMOT_OP_PRELU || op_id == MARMOT_OP_ABS ||
               op_id == MARMOT_OP_NEG || op_id == MARMOT_OP_SIGN || op_id == MARMOT_OP_SQRT || op_id == MARMOT_OP_EXP ||
               op_id == MARMOT_OP_LOG || op_id == MARMOT_OP_BITWISE_NOT) {
        ok = infer_elementwise_signature(values, node);
    } else if (op_id == MARMOT_OP_SOFTMAX || op_id == MARMOT_OP_LAYERNORM || op_id == MARMOT_OP_RMS_NORM ||
               op_id == MARMOT_OP_RMS_NORM_GEMMA || op_id == MARMOT_OP_REDUCTION_SUM ||
               op_id == MARMOT_OP_REDUCTION_MAX || op_id == MARMOT_OP_REDUCTION_MIN ||
               op_id == MARMOT_OP_REDUCTION_MEAN || op_id == MARMOT_OP_REDUCTION_PROD) {
        ok = infer_elementwise_signature(values, node);
    } else if (op_id == MARMOT_OP_PAGED_ATTENTION) {
        ok = infer_paged_attention_signature(values, node);
    } else {
        ok = infer_passthrough_signature(values, node);
    }

    if (!ok) {
        return false;
    }
    if (node.signature.stride_mode == MARMOT_STRIDE_MODE_ANY) {
        node.signature.stride_mode = infer_stride_mode(values, node);
    }
    return true;
}

} // namespace marmot::graph
