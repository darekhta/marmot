#include "fast_exec.hpp"

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/ops/quantization.h"
#include "marmot/traits_ids.gen.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core/bytecode/bytecode_compile.h"
#include "core/helpers/quant.h"
#include "execution_session.hpp"
#include "graph_executor.hpp"
#include "graph_node.hpp"
#include "kernel_dispatch_args.gen.h"

namespace {

bool fast_tensor_is_block_quantized_weight(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tensor->quant_kind);
    if (traits == nullptr || !traits->is_block_quantized) {
        return false;
    }
    if (!marmot_quant_storage_dtype_compatible(traits, tensor->dtype)) {
        return false;
    }
    return tensor->quant_layout == traits->layout;
}

extern "C" {
marmot_error_t cpu_matmul_quantized_with_hints(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, uint32_t hints
);
marmot_error_t cpu_matmul_quantized_q2_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q3_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q4_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q5_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q6_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q8_k(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q4_0(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q4_1(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q5_0(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q5_1(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q8_0(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
marmot_error_t cpu_matmul_quantized_q8_1(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
}

constexpr uint32_t CPU_QUANT_MATMUL_HINT_OUTPUT_PROJECTION = 1u << 0;

bool fast_exec_trace_enabled() {
    static bool enabled = [] {
        const char *env = std::getenv("MARMOT_GRAPH_TRACE");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool fast_exec_nan_check_enabled() {
    static bool enabled = [] {
        const char *env = std::getenv("MARMOT_GRAPH_NAN_CHECK");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

uint64_t fast_exec_now_ns() {
    using clock = std::chrono::steady_clock;
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
}

bool infer_rope_broadcast_params(const marmot_tensor_t *input, size_t &total_seqs_out, size_t &seq_len_out) {
    if (input == nullptr || input->shape.ndim < 2) {
        return false;
    }
    const size_t ndim = input->shape.ndim;
    const size_t dim = input->shape.shape[ndim - 1];
    const size_t seq_len = input->shape.shape[ndim - 2];
    if (dim == 0 || seq_len == 0) {
        return false;
    }
    const size_t total_tokens = marmot_tensor_num_elements(input);
    if (total_tokens % (seq_len * dim) != 0) {
        return false;
    }
    total_seqs_out = total_tokens / (seq_len * dim);
    seq_len_out = seq_len;
    return total_seqs_out > 0;
}

bool graph_node_is_qkv(const marmot::graph::GraphNode &node) {
    return node.signature.op_id == MARMOT_OP_QKV_SHARED_INPUT || node.signature.op_id == MARMOT_OP_QKV_PROJECTION ||
        node.signature.op_id == MARMOT_OP_QKV_ROPE;
}

bool graph_node_uses_qkv_rope(const marmot::graph::GraphNode &node) {
    return graph_node_is_qkv(node) && (node.signature.epilogue_flags & MARMOT_EPILOGUE_ROPE) != 0;
}

void restore_rope_binding(
    std::span<marmot_tensor_t *> bindings, marmot_tensor_t *&restore_ptr, marmot_value_id_t &restore_id
) {
    if (restore_ptr != nullptr && restore_id < bindings.size()) {
        bindings[restore_id] = restore_ptr;
    }
    restore_ptr = nullptr;
    restore_id = MARMOT_VALUE_ID_INVALID;
}

void init_profile(
    const marmot_context_t *ctx, const marmot::graph::FastPlan &plan, marmot::graph::FastExecProfile *profile
) {
    if (ctx == nullptr || profile == nullptr) {
        return;
    }
    profile->backend = ctx->backend_type;
    profile->phase = plan.phase();
    profile->total_ns = 0;
    profile->stages.clear();
    profile->stages.reserve(plan.stages().size());
    for (const marmot::graph::FastPlan::Stage &stage : plan.stages()) {
        profile->stages.push_back(
            marmot::graph::FastExecProfileStage{
                .kind = stage.kind,
                .first_node = stage.first_node,
                .node_count = stage.node_count,
                .executed_ops = 0,
                .duration_ns = 0,
            }
        );
    }
}

void record_stage_op(marmot::graph::FastExecProfileStage *profile_stage) {
    if (profile_stage != nullptr) {
        profile_stage->executed_ops++;
    }
}

template <typename TensorT = marmot_tensor_t>
[[nodiscard]] TensorT *lookup_binding(std::span<marmot_tensor_t *> bindings, marmot_value_id_t id) {
    if (id == MARMOT_VALUE_ID_INVALID || id >= bindings.size()) {
        return nullptr;
    }
    return static_cast<TensorT *>(bindings[id]);
}

[[nodiscard]] marmot_error_t dispatch_fast_op(
    const marmot_context_t *ctx, const marmot::graph::FastBytecodeOpRef &op, const void *args, const char *label
) {
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_bc_exec_ctx_t exec_ctx{
        .ctx = ctx,
        .device_ctx = ctx->device_ctx,
    };
    if (op.bc_op_index != MARMOT_BC_OP_INVALID) {
        return marmot_bc_execute_op(ctx->backend_type, op.bc_op_index, &exec_ctx, args);
    }

    marmot_bc_selection_t selection = marmot_bc_compile_signature(ctx, &op.signature);
    if (!selection.supported) {
        const char *reason = selection.reason != nullptr ? selection.reason : "bytecode not supported";
        char msg[192];
        std::snprintf(msg, sizeof(msg), "%s fast stage dispatch unavailable: %s", label, reason);
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, msg);
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return marmot_bc_execute_op(ctx->backend_type, selection.op_index, &exec_ctx, args);
}

[[nodiscard]] marmot_error_t dispatch_fast_cpu_quantized_matmul(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *output, uint32_t hints = 0
) {
    if (ctx == nullptr || ctx->device_ctx == nullptr || input == nullptr || weight == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast CPU quantized matmul arguments must not be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (hints != 0) {
        return cpu_matmul_quantized_with_hints(ctx->device_ctx, input, weight, epilogue, output, hints);
    }

    switch (weight->quant_kind) {
    case MARMOT_QUANT_KIND_Q2_K:
        return cpu_matmul_quantized_q2_k(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q3_K:
        return cpu_matmul_quantized_q3_k(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q4_K:
        return cpu_matmul_quantized_q4_k(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q5_K:
        return cpu_matmul_quantized_q5_k(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q6_K:
        return cpu_matmul_quantized_q6_k(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q8_K:
        return cpu_matmul_quantized_q8_k(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q4_0:
        return cpu_matmul_quantized_q4_0(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q4_1:
        return cpu_matmul_quantized_q4_1(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q5_0:
        return cpu_matmul_quantized_q5_0(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q5_1:
        return cpu_matmul_quantized_q5_1(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q8_0:
        return cpu_matmul_quantized_q8_0(ctx->device_ctx, input, weight, epilogue, output);
    case MARMOT_QUANT_KIND_Q8_1:
        return cpu_matmul_quantized_q8_1(ctx->device_ctx, input, weight, epilogue, output);
    default:
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Fast CPU quantized matmul format is not supported");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
}

[[nodiscard]] marmot_error_t execute_rms_norm_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastRmsNormOp &op, std::span<marmot_tensor_t *> bindings,
    float eps
) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    const marmot_tensor_t *weight = lookup_binding(bindings, op.weight);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input == nullptr || weight == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast RMS norm binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_kernel_args_rms_norm_t args{
        .ctx = ctx,
        .input = input,
        .weight = weight,
        .residual = nullptr,
        .output = output,
        .eps = eps,
    };
    return dispatch_fast_op(ctx, op.op, &args, "rms_norm");
}

[[nodiscard]] marmot_error_t execute_matmul_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastMatmulOp &op, std::span<marmot_tensor_t *> bindings,
    uint32_t quant_hints = 0
) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    const marmot_tensor_t *weight = lookup_binding(bindings, op.weight);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input == nullptr || weight == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast matmul binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *bias = nullptr;
    marmot_matmul_epilogue_t epilogue{};
    const marmot_matmul_epilogue_t *epilogue_ptr = nullptr;
    if (op.bias != MARMOT_VALUE_ID_INVALID) {
        bias = lookup_binding<const marmot_tensor_t>(bindings, op.bias);
        if (bias == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast matmul bias binding missing");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        epilogue.bias = bias;
        epilogue.enable_output_cast = false;
        epilogue.output_dtype = output->dtype;
        epilogue_ptr = &epilogue;
    }

    if ((op.op.signature.op_id == MARMOT_OP_MATMUL || op.op.signature.op_id == MARMOT_OP_MATMUL_BIAS ||
         op.op.signature.op_id == MARMOT_OP_MATMUL_BIAS_RELU || op.op.signature.op_id == MARMOT_OP_MATMUL_BIAS_GELU ||
         op.op.signature.op_id == MARMOT_OP_MATMUL_BIAS_SILU) &&
        ctx->backend_type == MARMOT_BACKEND_CPU && fast_tensor_is_block_quantized_weight(weight)) {
        return dispatch_fast_cpu_quantized_matmul(ctx, input, weight, epilogue_ptr, output, quant_hints);
    }

    marmot_kernel_args_matmul_t args{
        .ctx = ctx,
        .input = input,
        .weight = weight,
        .epilogue = epilogue_ptr,
        .output = output,
    };
    return dispatch_fast_op(ctx, op.op, &args, "matmul");
}

[[nodiscard]] marmot_error_t execute_unary_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastUnaryOp &op, std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast unary binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_kernel_args_unary_t args{
        .ctx = ctx,
        .input = input,
        .params = nullptr,
        .output = output,
    };
    return dispatch_fast_op(ctx, op.op, &args, "unary");
}

[[nodiscard]] marmot_error_t execute_binary_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastBinaryOp &op, std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *input_a = lookup_binding(bindings, op.input_a);
    const marmot_tensor_t *input_b = lookup_binding(bindings, op.input_b);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input_a == nullptr || input_b == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast binary binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_kernel_args_binary_t args{
        .ctx = ctx,
        .input_a = input_a,
        .input_b = input_b,
        .output = output,
    };
    return dispatch_fast_op(ctx, op.op, &args, "binary");
}

[[nodiscard]] marmot_error_t execute_softmax_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastSoftmaxOp &op, std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast softmax binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_kernel_args_softmax_t args{
        .ctx = ctx,
        .input = input,
        .output = output,
        .axis = op.axis,
    };
    return dispatch_fast_op(ctx, op.op, &args, "softmax");
}

[[nodiscard]] marmot_error_t execute_topk_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastTopkOp &op, std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    marmot_tensor_t *values_out = lookup_binding(bindings, op.values_out);
    marmot_tensor_t *indices_out = lookup_binding(bindings, op.indices_out);
    if (input == nullptr || values_out == nullptr || indices_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast TOPK binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    uint32_t k = op.k;
    if (k == 0 && values_out->shape.ndim != 0) {
        const size_t width = values_out->shape.shape[values_out->shape.ndim - 1];
        if (width > UINT32_MAX) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast TOPK width exceeds uint32");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        k = static_cast<uint32_t>(width);
    }
    marmot_kernel_args_topk_t args{
        .ctx = ctx,
        .input = input,
        .values_out = values_out,
        .indices_out = indices_out,
        .axis = op.axis,
        .k = k,
    };
    return dispatch_fast_op(ctx, op.op, &args, "topk");
}

[[nodiscard]] marmot_error_t execute_moe_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastMoeExpertsOp &op, std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *hidden_states = lookup_binding(bindings, op.hidden_states);
    const marmot_tensor_t *gate_exps = lookup_binding(bindings, op.gate_exps);
    const marmot_tensor_t *up_exps = lookup_binding(bindings, op.up_exps);
    const marmot_tensor_t *down_exps = lookup_binding(bindings, op.down_exps);
    const marmot_tensor_t *topk_ids = lookup_binding(bindings, op.topk_ids);
    const marmot_tensor_t *topk_weights = lookup_binding(bindings, op.topk_weights);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (hidden_states == nullptr || gate_exps == nullptr || up_exps == nullptr || down_exps == nullptr ||
        topk_ids == nullptr || topk_weights == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast MoE binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_kernel_args_moe_experts_t args{
        .ctx = ctx,
        .hidden_states = hidden_states,
        .gate_exps = gate_exps,
        .up_exps = up_exps,
        .down_exps = down_exps,
        .topk_ids = topk_ids,
        .topk_weights = topk_weights,
        .out = output,
        .ffn_type = op.ffn_type,
        .weights_scale = op.weights_scale,
        .router_weight_policy = op.router_weight_policy,
    };
    return dispatch_fast_op(ctx, op.op, &args, "moe_experts");
}

[[nodiscard]] marmot_error_t execute_gather_rows_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastGatherRowsOp &op, std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    const marmot_tensor_t *indices = lookup_binding(bindings, op.indices);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input == nullptr || indices == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast gather_rows binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_kernel_args_gather_rows_t args{
        .ctx = ctx,
        .input = input,
        .indices = indices,
        .output = output,
    };
    return dispatch_fast_op(ctx, op.op, &args, "gather_rows");
}

[[nodiscard]] bool can_skip_identity_gather_rows(
    const marmot_context_t *ctx, const marmot::graph::FastGatherRowsOp &op, std::span<marmot_tensor_t *> bindings
) {
    if (ctx == nullptr || ctx->backend_type != MARMOT_BACKEND_CPU) {
        return false;
    }
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    const marmot_tensor_t *indices = lookup_binding(bindings, op.indices);
    if (input == nullptr || indices == nullptr || op.output == MARMOT_VALUE_ID_INVALID ||
        op.output >= bindings.size()) {
        return false;
    }
    if (input->shape.ndim == 0 || indices->shape.ndim == 0) {
        return false;
    }
    const size_t row_count = input->shape.shape[0];
    const size_t index_count = marmot_tensor_num_elements(indices);
    if (row_count == 0 || row_count != index_count) {
        return false;
    }
    const marmot_uint32_t *index_data = marmot_tensor_data_u32(ctx, const_cast<marmot_tensor_t *>(indices));
    if (index_data == nullptr) {
        return false;
    }
    for (size_t i = 0; i < index_count; ++i) {
        if (index_data[i].value != i) {
            return false;
        }
    }
    bindings[op.output] = const_cast<marmot_tensor_t *>(input);
    return true;
}

[[nodiscard]] marmot_error_t execute_vec_dot_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastVecDotOp &op, std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    const marmot_tensor_t *weight = lookup_binding(bindings, op.weight);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input == nullptr || weight == nullptr || output == nullptr || output->data == nullptr ||
        output->dtype != MARMOT_DTYPE_FLOAT32 || marmot_tensor_num_elements(output) != 1) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast vec_dot binding missing or invalid");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_vec_dot_descriptor_t desc{};
    if (marmot_vec_dot_descriptor_from_tensors(input, weight, &desc) != MARMOT_SUCCESS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast vec_dot descriptor inference failed");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_kernel_args_vec_dot_t args{
        .ctx = ctx,
        .desc = &desc,
        .result = static_cast<float *>(output->data),
    };
    return dispatch_fast_op(ctx, op.op, &args, "vec_dot");
}

[[nodiscard]] marmot_error_t
execute_reshape_supernode(const marmot::graph::FastReshapeOp &op, std::span<marmot_tensor_t *> bindings) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast reshape binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t input_elems = marmot_tensor_num_elements(input);
    size_t output_elems = 1;
    for (uint32_t i = 0; i < op.output_desc.ndim; ++i) {
        output_elems *= op.output_desc.shape[i];
    }
    if (input_elems != output_elems) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast reshape element count mismatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (output->owns_data && output->data != nullptr && output->data != input->data) {
        free(output->data);
    }

    output->shape.ndim = op.output_desc.ndim;
    for (uint32_t i = 0; i < op.output_desc.ndim; ++i) {
        output->shape.shape[i] = op.output_desc.shape[i];
        output->shape.strides[i] = op.output_desc.strides[i];
    }
    output->dtype = op.output_desc.dtype;
    output->data = input->data;
    output->capacity_bytes = input->capacity_bytes;
    output->owns_data = false;
    output->quant_params = input->quant_params;
    output->quant_kind = input->quant_kind;
    output->quant_layout = input->quant_layout;
    output->backend = input->backend;
    output->memory_location = input->memory_location;
    output->needs_sync = input->needs_sync;
    output->packed_data = nullptr;
    output->packed_src_data = nullptr;
    output->packed_bytes = 0;
    output->packed_row_bytes = 0;
    output->packed_rows = 0;
    return MARMOT_SUCCESS;
}

[[nodiscard]] marmot_error_t execute_rope_supernode(
    marmot::graph::ExecutionSession *session, const marmot_context_t *ctx, const marmot::graph::FastRopeOp &op,
    std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *input = lookup_binding(bindings, op.input);
    marmot_tensor_t *positions = lookup_binding(bindings, op.positions);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (input == nullptr || positions == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast RoPE binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_rope_params_t rope_params = marmot_rope_params_default();
    if (session != nullptr) {
        rope_params = *session->rope_params();
        size_t total_seqs = 0;
        size_t seq_len = 0;
        if (!infer_rope_broadcast_params(input, total_seqs, seq_len)) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fast RoPE input shape is not compatible");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        marmot_tensor_t *broadcast = session->broadcast_rope_positions(positions, total_seqs, seq_len);
        if (broadcast == nullptr) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fast RoPE positions broadcast failed");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        positions = broadcast;
    }
    rope_params.positions = positions;

    marmot_kernel_args_rope_t args{
        .ctx = ctx,
        .input = input,
        .output = output,
        .rope_params = &rope_params,
        .n_past = 0,
        .n_rot = 0,
    };
    return dispatch_fast_op(ctx, op.op, &args, "rope");
}

[[nodiscard]] marmot_error_t execute_paged_attention_supernode(
    const marmot_context_t *ctx, const marmot::graph::FastPagedAttentionOp &op, std::span<marmot_tensor_t *> bindings
) {
    const marmot_tensor_t *token_meta = lookup_binding(bindings, op.token_meta);
    const marmot_tensor_t *q = lookup_binding(bindings, op.q);
    const marmot_tensor_t *k = lookup_binding(bindings, op.k);
    const marmot_tensor_t *v = lookup_binding(bindings, op.v);
    marmot_tensor_t *kv_k = lookup_binding(bindings, op.kv_k);
    marmot_tensor_t *kv_v = lookup_binding(bindings, op.kv_v);
    const marmot_tensor_t *block_table = lookup_binding(bindings, op.block_table);
    marmot_tensor_t *kv_k_scale = lookup_binding(bindings, op.kv_k_scale);
    marmot_tensor_t *kv_v_scale = lookup_binding(bindings, op.kv_v_scale);
    marmot_tensor_t *output = lookup_binding(bindings, op.output);
    if (token_meta == nullptr || q == nullptr || k == nullptr || v == nullptr || kv_k == nullptr || kv_v == nullptr ||
        block_table == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast paged attention binding missing");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t token_count = token_meta->shape.shape[0];
    const size_t num_q_heads = q->shape.shape[1];
    const size_t head_dim = q->shape.shape[2];
    const size_t num_kv_heads = k->shape.shape[1];
    const size_t block_size = kv_k->shape.shape[3];
    const float scale = head_dim > 0 ? 1.0f / std::sqrt((float)head_dim) : 1.0f;

    marmot_kernel_args_paged_attention_t args{
        .ctx = ctx,
        .token_meta = token_meta,
        .q = q,
        .k_new = k,
        .v_new = v,
        .kv_k = kv_k,
        .kv_v = kv_v,
        .block_table = block_table,
        .kv_k_scale = kv_k_scale,
        .kv_v_scale = kv_v_scale,
        .out = output,
        .token_count = static_cast<uint32_t>(token_count),
        .layer_idx = op.layer_idx,
        .num_q_heads = static_cast<uint32_t>(num_q_heads),
        .num_kv_heads = static_cast<uint32_t>(num_kv_heads),
        .head_dim = static_cast<uint32_t>(head_dim),
        .block_size = static_cast<uint32_t>(block_size),
        .scale = scale,
    };
    return dispatch_fast_op(ctx, op.op, &args, "paged_attention");
}

[[nodiscard]] marmot_error_t execute_attention_decode_supernode(
    marmot::graph::ExecutionSession *session, const marmot_context_t *ctx,
    const marmot::graph::FastAttentionDecodeSupernode &supernode, std::span<marmot_tensor_t *> bindings,
    float rms_norm_eps, marmot::graph::FastExecProfileStage *profile_stage
) {
    auto exec = [&](auto &&fn) -> marmot_error_t {
        const marmot_error_t status = fn();
        if (status == MARMOT_SUCCESS) {
            record_stage_op(profile_stage);
        }
        return status;
    };

    marmot_error_t status =
        exec([&] { return execute_rms_norm_supernode(ctx, supernode.attn_norm, bindings, rms_norm_eps); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_matmul_supernode(ctx, supernode.q_proj, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_matmul_supernode(ctx, supernode.k_proj, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_matmul_supernode(ctx, supernode.v_proj, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    if (supernode.q_norm_reshape.has_value()) {
        status = exec([&] { return execute_reshape_supernode(*supernode.q_norm_reshape, bindings); });
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }
    if (supernode.q_norm.has_value()) {
        status = exec([&] { return execute_rms_norm_supernode(ctx, *supernode.q_norm, bindings, rms_norm_eps); });
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }
    if (supernode.k_norm_reshape.has_value()) {
        status = exec([&] { return execute_reshape_supernode(*supernode.k_norm_reshape, bindings); });
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }
    if (supernode.k_norm.has_value()) {
        status = exec([&] { return execute_rms_norm_supernode(ctx, *supernode.k_norm, bindings, rms_norm_eps); });
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }

    status = exec([&] { return execute_reshape_supernode(supernode.q_heads, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_reshape_supernode(supernode.k_heads, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_reshape_supernode(supernode.v_heads, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_rope_supernode(session, ctx, supernode.q_rope, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_rope_supernode(session, ctx, supernode.k_rope, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_reshape_supernode(supernode.q_tokens, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_reshape_supernode(supernode.k_tokens, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_reshape_supernode(supernode.v_tokens, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_paged_attention_supernode(ctx, supernode.paged_attention, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_reshape_supernode(supernode.attn_flat, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    status = exec([&] { return execute_matmul_supernode(ctx, supernode.out_proj, bindings); });
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return exec([&] { return execute_binary_supernode(ctx, supernode.residual_add, bindings); });
}

} // namespace

namespace marmot::graph {

bool FastExec::supports(const marmot_context_t *ctx, const FastPlan &plan) noexcept {
    return ctx != nullptr && plan.phase() == FastPlanPhase::Decode && !fast_exec_trace_enabled() &&
        !fast_exec_nan_check_enabled();
}

marmot_error_t FastExec::prepare_node_rope(
    ExecutionSession *session, Executor &executor, const GraphNode &node, std::span<marmot_tensor_t *> bindings,
    marmot_tensor_t *&restore_ptr, marmot_value_id_t &restore_id
) {
    restore_ptr = nullptr;
    restore_id = MARMOT_VALUE_ID_INVALID;
    if (session == nullptr) {
        return MARMOT_SUCCESS;
    }

    if (node.signature.op_id == MARMOT_OP_ROPE && node.inputs.size() >= 2) {
        const marmot_value_id_t input_id = node.inputs[0];
        const marmot_value_id_t positions_id = node.inputs[1];
        if (input_id >= bindings.size() || positions_id >= bindings.size()) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE bindings out of range");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        marmot_tensor_t *input = bindings[input_id];
        marmot_tensor_t *positions = bindings[positions_id];
        if (input == nullptr || positions == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE input or positions not bound");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        size_t total_seqs = 0;
        size_t seq_len = 0;
        if (!infer_rope_broadcast_params(input, total_seqs, seq_len)) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE input shape is not compatible");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        marmot_tensor_t *broadcast = session->broadcast_rope_positions(positions, total_seqs, seq_len);
        if (broadcast == nullptr) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE positions broadcast failed");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (broadcast != positions) {
            restore_ptr = positions;
            restore_id = positions_id;
            bindings[positions_id] = broadcast;
        }
        return MARMOT_SUCCESS;
    }

    if (!graph_node_uses_qkv_rope(node)) {
        return MARMOT_SUCCESS;
    }
    if (node.inputs.size() < 5) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE node missing positions input");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_value_id_t input_id = node.inputs[0];
    const marmot_value_id_t positions_id = node.inputs.back();
    if (input_id >= bindings.size() || positions_id >= bindings.size()) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE bindings out of range");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_tensor_t *input = bindings[input_id];
    marmot_tensor_t *positions = bindings[positions_id];
    if (input == nullptr || positions == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV RoPE input or positions not bound");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t total_seqs = 0;
    size_t seq_len = 0;
    if (!infer_rope_broadcast_params(input, total_seqs, seq_len)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV RoPE input shape is not compatible");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    marmot_tensor_t *broadcast = session->broadcast_rope_positions(positions, total_seqs, seq_len);
    if (broadcast == nullptr) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV RoPE positions broadcast failed");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (broadcast != positions) {
        restore_ptr = positions;
        restore_id = positions_id;
        bindings[positions_id] = broadcast;
    }

    return executor.update_qkv_rope_params(bindings);
}

marmot_error_t FastExec::execute(
    Graph::Impl &impl, ExecutionSession *session, const FastPlan &plan, const marmot_context_t *ctx,
    std::span<marmot_tensor_t *> bindings, FastExecProfile *profile
) {
    if (ctx == nullptr || session == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "FastExec requires context and session");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    init_profile(ctx, plan, profile);
    const uint64_t total_start_ns = profile != nullptr ? fast_exec_now_ns() : 0;

    marmot_error_t batch_begin = marmot_graph_batch_begin(ctx);
    if (batch_begin != MARMOT_SUCCESS) {
        return batch_begin;
    }

    Executor executor(impl, session);
    marmot_error_t status = MARMOT_SUCCESS;

    for (size_t stage_index = 0; stage_index < plan.stages().size() && status == MARMOT_SUCCESS; ++stage_index) {
        const FastPlan::Stage &stage = plan.stages()[stage_index];
        const uint64_t stage_start_ns = profile != nullptr ? fast_exec_now_ns() : 0;
        FastExecProfileStage *profile_stage =
            (profile != nullptr && stage_index < profile->stages.size()) ? &profile->stages[stage_index] : nullptr;
        status = execute_stage(impl, session, executor, stage, ctx, bindings, profile_stage);

        if (profile != nullptr && stage_index < profile->stages.size()) {
            const uint64_t stage_end_ns = fast_exec_now_ns();
            profile->stages[stage_index].duration_ns += stage_end_ns - stage_start_ns;
        }
    }

    marmot_error_t batch_end = marmot_graph_batch_end(ctx, status == MARMOT_SUCCESS);
    if (status == MARMOT_SUCCESS && profile != nullptr) {
        profile->total_ns = fast_exec_now_ns() - total_start_ns;
    }
    if (status == MARMOT_SUCCESS) {
        return batch_end;
    }
    return status;
}

marmot_error_t FastExec::execute_stage(
    Graph::Impl &impl, ExecutionSession *session, Executor &executor, const FastPlan::Stage &stage,
    const marmot_context_t *ctx, std::span<marmot_tensor_t *> bindings, FastExecProfileStage *profile_stage
) {
    auto execute_fallback = [&]() -> marmot_error_t {
        for (uint32_t offset = 0; offset < stage.node_count; ++offset) {
            const uint32_t node_index = stage.first_node + offset;
            if (node_index >= impl.nodes.size()) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fast plan node range exceeds graph");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }

            const GraphNode &node = impl.nodes[node_index];
            if (node.skip) {
                continue;
            }

            marmot_tensor_t *restore_ptr = nullptr;
            marmot_value_id_t restore_id = MARMOT_VALUE_ID_INVALID;
            marmot_error_t status = prepare_node_rope(session, executor, node, bindings, restore_ptr, restore_id);
            if (status != MARMOT_SUCCESS) {
                restore_rope_binding(bindings, restore_ptr, restore_id);
                return status;
            }

            status = executor.execute_node(node_index, node, ctx, bindings);
            restore_rope_binding(bindings, restore_ptr, restore_id);
            if (status != MARMOT_SUCCESS) {
                return status;
            }
            record_stage_op(profile_stage);
        }
        return MARMOT_SUCCESS;
    };

    const float rms_norm_eps = session != nullptr ? session->rms_norm_eps() : 1e-6f;

    if (const auto *payload = std::get_if<FastAttentionDecodeSupernode>(&stage.payload)) {
        return execute_attention_decode_supernode(session, ctx, *payload, bindings, rms_norm_eps, profile_stage);
    }

    if (const auto *payload = std::get_if<FastDenseFfnGeluSupernode>(&stage.payload)) {
        marmot_error_t status = execute_rms_norm_supernode(ctx, payload->rms_norm, bindings, rms_norm_eps);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_matmul_supernode(ctx, payload->up_proj, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_unary_supernode(ctx, payload->gelu, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_matmul_supernode(ctx, payload->down_proj, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_binary_supernode(ctx, payload->residual_add, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);
        return MARMOT_SUCCESS;
    }

    if (const auto *payload = std::get_if<FastDenseFfnGatedSupernode>(&stage.payload)) {
        marmot_error_t status = execute_rms_norm_supernode(ctx, payload->rms_norm, bindings, rms_norm_eps);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_matmul_supernode(ctx, payload->gate_proj, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_matmul_supernode(ctx, payload->up_proj, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_binary_supernode(ctx, payload->glu, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_matmul_supernode(ctx, payload->down_proj, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_binary_supernode(ctx, payload->residual_add, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);
        return MARMOT_SUCCESS;
    }

    if (const auto *payload = std::get_if<FastMoeFfnBasicSupernode>(&stage.payload)) {
        marmot_error_t status = execute_rms_norm_supernode(ctx, payload->rms_norm, bindings, rms_norm_eps);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_matmul_supernode(ctx, payload->router, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        if (payload->topk_first) {
            status = execute_topk_supernode(ctx, payload->topk, bindings);
            if (status != MARMOT_SUCCESS)
                return status;
            record_stage_op(profile_stage);

            status = execute_softmax_supernode(ctx, payload->softmax, bindings);
            if (status != MARMOT_SUCCESS)
                return status;
            record_stage_op(profile_stage);
        } else {
            status = execute_softmax_supernode(ctx, payload->softmax, bindings);
            if (status != MARMOT_SUCCESS)
                return status;
            record_stage_op(profile_stage);

            status = execute_topk_supernode(ctx, payload->topk, bindings);
            if (status != MARMOT_SUCCESS)
                return status;
            record_stage_op(profile_stage);
        }

        status = execute_moe_supernode(ctx, payload->moe, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);

        status = execute_binary_supernode(ctx, payload->residual_add, bindings);
        if (status != MARMOT_SUCCESS)
            return status;
        record_stage_op(profile_stage);
        return MARMOT_SUCCESS;
    }

    if (const auto *payload = std::get_if<FastLogitsSupernode>(&stage.payload)) {
        marmot_error_t status = MARMOT_SUCCESS;
        if (payload->final_norm.has_value()) {
            status = execute_rms_norm_supernode(ctx, *payload->final_norm, bindings, rms_norm_eps);
            if (status != MARMOT_SUCCESS)
                return status;
            record_stage_op(profile_stage);
        }
        if (payload->gather.has_value()) {
            if (!can_skip_identity_gather_rows(ctx, *payload->gather, bindings)) {
                status = execute_gather_rows_supernode(ctx, *payload->gather, bindings);
                if (status != MARMOT_SUCCESS)
                    return status;
            }
            record_stage_op(profile_stage);
        }
        if (payload->matmul.has_value()) {
            status = execute_matmul_supernode(ctx, *payload->matmul, bindings, CPU_QUANT_MATMUL_HINT_OUTPUT_PROJECTION);
            if (status != MARMOT_SUCCESS)
                return status;
            record_stage_op(profile_stage);
            return MARMOT_SUCCESS;
        }
        if (payload->vec_dot.has_value()) {
            status = execute_vec_dot_supernode(ctx, *payload->vec_dot, bindings);
            if (status != MARMOT_SUCCESS)
                return status;
            record_stage_op(profile_stage);
            return MARMOT_SUCCESS;
        }
    }

    return execute_fallback();
}

} // namespace marmot::graph
